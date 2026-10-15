from threading import Barrier
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlsplit

from fastapi.testclient import TestClient
from httpx import TimeoutException
import pytest

from api import index

client = TestClient(index.app)


def test_invalid_payload_is_rejected_without_calling_gemini(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("Gemini must not be called for an invalid payload.")

    monkeypatch.setattr(index.genai, "Client", fail_if_called)

    response = client.post(
        "/api/recommend",
        json={
            "content_type": "ambos",
            "genres": [],
            "preferences": "demasiado corto",
        },
    )

    assert response.status_code == 422


def test_recommendation_request_uses_server_key_and_returns_schema(monkeypatch):
    class FakeModels:
        def generate_content(self, **kwargs):
            assert kwargs["model"] == "gemini-test-model"
            assert "<user_data>" in kwargs["contents"]
            return type(
                "GeminiResult",
                (),
                {
                    "text": (
                        '{"recommendations": ['
                        '{"type":"pelicula","title":"Pelicula 1","creator":"Directora",'
                        '"synopsis":"Una historia de misterio y amistad.",'
                        '"reason":"Coincide con tus géneros."},'
                        '{"type":"libro","title":"Libro 1","creator":"Autora",'
                        '"synopsis":"Una novela de aventuras y misterio.",'
                        '"reason":"Encaja con lo que buscas."},'
                        '{"type":"serie","title":"Serie 2","creator":"Creadora",'
                        '"synopsis":"Una aventura fantástica para toda la familia.",'
                        '"reason":"Incluye fantasía y aventura."}'
                        "]}"
                    )
                },
            )()

    class FakeClient:
        def __init__(self, api_key, **kwargs):
            assert api_key == "test-server-key"
            assert kwargs["http_options"].timeout == 30_000
            self.models = FakeModels()

    monkeypatch.setenv("GEMINI_API_KEY", "test-server-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-test-model")
    monkeypatch.setenv("TMDB_API_KEY", "test-tmdb-key")
    monkeypatch.setenv("GOOGLE_BOOKS_API_KEY", "test-books-server-key")
    monkeypatch.setattr(index.genai, "Client", FakeClient)
    monkeypatch.setattr(
        index,
        "_lookup_screen_title",
        lambda recommendation, api_key, region: {
            "image_url": "https://image.example/poster.jpg",
            "platforms": ["Netflix"],
            "availability_url": "https://www.themoviedb.org/",
        },
    )
    received_book_keys = []

    def fake_lookup_book(recommendation, api_key):
        received_book_keys.append(api_key)
        return {}, set()

    monkeypatch.setattr(index, "_lookup_book", fake_lookup_book)

    response = client.post(
        "/api/recommend",
        json={
            "content_type": "ambos",
            "genres": ["misterio", "fantasía"],
            "preferences": "Me gustan las historias de misterio y aventura.",
        },
    )

    assert response.status_code == 200
    assert len(response.json()["recommendations"]) == 3
    assert {item["type"] for item in response.json()["recommendations"]} == {
        "pelicula",
        "libro",
        "serie",
    }
    assert response.json()["recommendations"][0]["platforms"] == ["Netflix"]
    assert received_book_keys == ["test-books-server-key"]
    assert "test-books-server-key" not in response.text


def test_screen_title_lookup_returns_poster_and_spanish_platforms(monkeypatch):
    responses = iter(
        [
            {
                "results": [
                    {
                        "id": 42,
                        "name": "Serie de prueba",
                        "poster_path": "/poster.jpg",
                    }
                ]
            },
            {
                "results": {
                    "ES": {
                        "link": "https://www.themoviedb.org/tv/42",
                        "flatrate": [{"provider_name": "Netflix"}],
                        "rent": [{"provider_name": "Prime Video"}],
                    }
                }
            },
        ]
    )
    monkeypatch.setattr(index, "_fetch_json", lambda *args: next(responses))
    recommendation = index.Recommendation(
        type="serie",
        title="Serie de prueba",
        creator="Creador",
        synopsis="Sinopsis de prueba.",
        reason="Coincide.",
    )

    metadata = index._lookup_screen_title(recommendation, "test-key", "ES")

    assert metadata["image_url"] == "https://image.tmdb.org/t/p/w500/poster.jpg"
    assert metadata["platforms"] == ["Netflix", "Prime Video"]
    assert metadata["availability_url"] == "https://www.themoviedb.org/tv/42"


def test_book_lookup_returns_google_books_cover_and_author(monkeypatch):
    monkeypatch.setattr(
        index,
        "_fetch_json",
        lambda *args: {
            "items": [
                {
                    "volumeInfo": {
                        "title": "Libro de prueba",
                        "authors": ["Autora de prueba"],
                        "imageLinks": {"thumbnail": "http://books.example/cover.jpg"},
                    }
                }
            ]
        },
    )
    recommendation = index.Recommendation(
        type="libro",
        title="Libro de prueba",
        creator="Autora",
        synopsis="Sinopsis de prueba.",
        reason="Coincide.",
    )

    book_data, rate_limited = index._lookup_book(recommendation)

    assert book_data["image_url"] == "https://books.example/cover.jpg"
    assert book_data["creator"] == "Autora de prueba"
    assert rate_limited == set()


def test_book_lookup_uses_server_api_key(monkeypatch):
    observed = {}

    def fake_fetch_json(url, provider, operation):
        observed["provider"] = provider
        observed["query"] = parse_qs(urlsplit(url).query)
        return {
            "items": [
                {
                    "volumeInfo": {
                        "title": "Berserk",
                        "imageLinks": {"thumbnail": "https://books.example/cover.jpg"},
                    }
                }
            ]
        }

    monkeypatch.setattr(index, "_fetch_json", fake_fetch_json)
    recommendation = index.Recommendation(
        type="libro",
        title="Berserk",
        creator="Kentaro Miura",
        synopsis="Manga.",
        reason="Coincide.",
    )

    index._lookup_book(recommendation, "server-only-books-key")

    assert observed["provider"] == "Google Books"
    assert observed["query"]["key"] == ["server-only-books-key"]


@pytest.mark.parametrize("google_books_rate_limited", [True, False])
def test_book_cover_falls_back_to_open_library(
    google_books_rate_limited, monkeypatch
):
    def fake_fetch_json(url, provider, operation):
        if provider == "Google Books":
            if google_books_rate_limited:
                raise index.CatalogRateLimited("Google Books")
            return {
                "items": [
                    {
                        "volumeInfo": {
                            "title": "Berserk",
                            "authors": ["Kentaro Miura"],
                            "imageLinks": {},
                        }
                    }
                ]
            }
        assert provider == "Open Library"
        return {
            "docs": [
                {
                    "title": "Berserk",
                    "author_name": ["Kentaro Miura"],
                    "cover_i": 12345,
                }
            ]
        }

    monkeypatch.setattr(index, "_fetch_json", fake_fetch_json)
    recommendation = index.Recommendation(
        type="libro",
        title="Berserk",
        creator="Kentaro Miura",
        synopsis="Manga de fantasía oscura.",
        reason="Coincide.",
    )

    book_data, rate_limited = index._lookup_book(recommendation)

    assert book_data["image_url"] == "https://covers.openlibrary.org/b/id/12345-L.jpg"
    if google_books_rate_limited:
        assert rate_limited == {"Google Books"}
    else:
        assert rate_limited == set()


def test_open_library_retries_without_author_when_no_cover_matches(monkeypatch):
    open_library_searches = []

    def fake_fetch_json(url, provider, operation):
        if provider == "Google Books":
            return {"items": []}
        query = parse_qs(urlsplit(url).query)
        open_library_searches.append(query)
        if "author" in query:
            return {"docs": [{"title": "Berserk"}]}
        return {
            "docs": [
                {
                    "title": "Berserk",
                    "author_name": ["Kentaro Miura"],
                    "cover_i": 12345,
                }
            ]
        }

    monkeypatch.setattr(index, "_fetch_json", fake_fetch_json)
    recommendation = index.Recommendation(
        type="libro",
        title="Berserk",
        creator="Autor sugerido que no coincide",
        synopsis="Manga.",
        reason="Coincide.",
    )

    book_data, _ = index._lookup_book(recommendation)

    assert len(open_library_searches) == 2
    assert "author" in open_library_searches[0]
    assert "author" not in open_library_searches[1]
    assert book_data["image_url"] == "https://covers.openlibrary.org/b/id/12345-L.jpg"


def test_google_books_503_falls_back_to_open_library(monkeypatch, caplog):
    providers_called = []

    def fake_fetch_json(url, provider, operation):
        providers_called.append(provider)
        if provider == "Google Books":
            raise index.HTTPException(
                status_code=502,
                detail="Google Books responded with HTTP 503.",
            )
        return {
            "docs": [
                {
                    "title": "Berserk",
                    "author_name": ["Kentaro Miura"],
                    "cover_i": 12345,
                }
            ]
        }

    monkeypatch.setattr(index, "_fetch_json", fake_fetch_json)
    recommendation = index.Recommendation(
        type="libro",
        title="Berserk",
        creator="Kentaro Miura",
        synopsis="Manga.",
        reason="Coincide.",
    )

    book_data, unavailable_providers = index._lookup_book(
        recommendation,
        "test-books-api-key",
    )

    assert book_data["image_url"] == "https://covers.openlibrary.org/b/id/12345-L.jpg"
    assert unavailable_providers == {"Google Books"}
    assert providers_called == ["Google Books", "Open Library"]
    assert "fallback_provider=Open Library" in caplog.text


def test_recommendation_metadata_is_fetched_concurrently(monkeypatch):
    barrier = Barrier(3)
    recommendations = [
        index.Recommendation(
            type="serie",
            title=f"Serie {number}",
            creator="Creador",
            synopsis="Sinopsis.",
            reason="Coincide.",
        )
        for number in range(3)
    ]

    def wait_for_other_catalog_requests(recommendation, *args):
        barrier.wait(timeout=1)
        return recommendation, set()

    monkeypatch.setattr(
        index,
        "_enrich_recommendation",
        wait_for_other_catalog_requests,
    )

    enriched, rate_limited = index._enrich_recommendations(
        recommendations,
        "test-key",
        "ES",
    )

    assert rate_limited == set()
    assert [item.title for item in enriched] == [
        "Serie 0",
        "Serie 1",
        "Serie 2",
    ]


def test_catalog_rate_limit_preserves_recommendations(monkeypatch):
    recommendations = [
        index.Recommendation(
            type="serie",
            title="Serie recomendada",
            creator="Creador",
            synopsis="Una historia de ejemplo.",
            reason="Coincide con tus gustos.",
        )
        for _ in range(3)
    ]

    def raise_rate_limit(url, timeout):
        raise HTTPError(url, 429, "Too Many Requests", {}, None)

    monkeypatch.setattr(index, "urlopen", raise_rate_limit)
    enriched, rate_limited = index._enrich_recommendations(
        recommendations,
        "test-key",
        "ES",
    )

    assert rate_limited == {"TMDb"}
    assert enriched == recommendations


@pytest.mark.parametrize("provider", ["Google Books", "TMDb"])
def test_catalog_rate_limit_identifies_provider(provider, monkeypatch, caplog):
    def raise_rate_limit(url, timeout):
        raise HTTPError(url, 429, "Too Many Requests", {}, None)

    monkeypatch.setattr(index, "urlopen", raise_rate_limit)

    with pytest.raises(index.CatalogRateLimited) as error:
        index._fetch_json("https://catalog.example/search", provider, "test search")

    assert error.value.provider == provider
    assert f"provider={provider}" in caplog.text
    assert "status=429" in caplog.text
    assert "catalog.example" not in caplog.text


def test_catalog_http_error_identifies_provider_without_logging_url(
    monkeypatch, caplog
):
    def raise_forbidden(url, timeout):
        raise HTTPError(url, 403, "Forbidden", {}, None)

    monkeypatch.setattr(index, "urlopen", raise_forbidden)

    with pytest.raises(index.HTTPException) as error:
        index._fetch_json(
            "https://catalog.example/search?secret=hidden",
            "Google Books",
            "búsqueda de libros",
        )

    assert error.value.status_code == 502
    assert "Google Books respondió con HTTP 403" in error.value.detail
    assert "provider=Google Books" in caplog.text
    assert "catalog.example" not in caplog.text
    assert "secret=hidden" not in caplog.text


def test_screen_recommendations_require_tmdb_key(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("Gemini must not be called without the TMDb key.")

    monkeypatch.setenv("GEMINI_API_KEY", "test-server-key")
    monkeypatch.delenv("TMDB_API_KEY", raising=False)
    monkeypatch.setattr(index.genai, "Client", fail_if_called)

    response = client.post(
        "/api/recommend",
        json={
            "content_type": "serie",
            "genres": ["misterio"],
            "preferences": "Me gustan las series de misterio.",
        },
    )

    assert response.status_code == 503
    assert "TMDB_API_KEY" in response.json()["detail"]


def test_gemini_overload_retries_and_returns_service_unavailable(monkeypatch):
    class FakeAPIError(Exception):
        def __init__(self, code, message):
            self.code = code
            self.message = message

    class FakeModels:
        def generate_content(self, **kwargs):
            self.attempts += 1
            raise FakeAPIError(503, "temporarily overloaded")

    fake_models = FakeModels()
    fake_models.attempts = 0

    class FakeClient:
        def __init__(self, api_key, **kwargs):
            self.models = fake_models

    monkeypatch.setenv("GEMINI_API_KEY", "test-server-key")
    monkeypatch.setenv("TMDB_API_KEY", "test-tmdb-key")
    monkeypatch.setattr(index.errors, "APIError", FakeAPIError)
    monkeypatch.setattr(index.genai, "Client", FakeClient)
    monkeypatch.setattr(index.time, "sleep", lambda _: None)

    response = client.post(
        "/api/recommend",
        json={
            "content_type": "ambos",
            "genres": ["misterio"],
            "preferences": "Me gustan las historias de misterio.",
        },
    )

    assert response.status_code == 503
    assert "temporalmente saturado" in response.json()["detail"]
    assert fake_models.attempts == 3


def test_gemini_quota_exceeded_returns_clear_429_without_retry(monkeypatch):
    class FakeAPIError(Exception):
        def __init__(self, code, message):
            self.code = code
            self.message = message

    class FakeModels:
        def __init__(self):
            self.attempts = 0

        def generate_content(self, **kwargs):
            self.attempts += 1
            raise FakeAPIError(
                429,
                "You exceeded your current quota. Please retry in 13h11m52.16s.",
            )

    fake_models = FakeModels()

    class FakeClient:
        def __init__(self, api_key, **kwargs):
            self.models = fake_models

    monkeypatch.setenv("GEMINI_API_KEY", "test-server-key")
    monkeypatch.setenv("TMDB_API_KEY", "test-tmdb-key")
    monkeypatch.setattr(index.errors, "APIError", FakeAPIError)
    monkeypatch.setattr(index.genai, "Client", FakeClient)

    response = client.post(
        "/api/recommend",
        json={
            "content_type": "serie",
            "genres": ["misterio"],
            "preferences": "Me gustan las series de misterio.",
        },
    )

    assert response.status_code == 429
    assert "Se ha agotado la cuota" in response.json()["detail"]
    assert "13h11m52.16s" in response.json()["detail"]
    assert "ai.dev/rate-limit" in response.json()["detail"]
    assert fake_models.attempts == 1


def test_gemini_deadline_retries_and_returns_gateway_timeout(monkeypatch):
    class FakeAPIError(Exception):
        def __init__(self, code, message):
            self.code = code
            self.message = message

    class FakeModels:
        def __init__(self):
            self.attempts = 0

        def generate_content(self, **kwargs):
            self.attempts += 1
            raise FakeAPIError(504, "Deadline expired before operation could complete.")

    fake_models = FakeModels()

    class FakeClient:
        def __init__(self, api_key, **kwargs):
            self.models = fake_models

    monkeypatch.setenv("GEMINI_API_KEY", "test-server-key")
    monkeypatch.setenv("TMDB_API_KEY", "test-tmdb-key")
    monkeypatch.setattr(index.errors, "APIError", FakeAPIError)
    monkeypatch.setattr(index.genai, "Client", FakeClient)
    monkeypatch.setattr(index.time, "sleep", lambda _: None)

    response = client.post(
        "/api/recommend",
        json={
            "content_type": "serie",
            "genres": ["misterio"],
            "preferences": "Me gustan las series de misterio.",
        },
    )

    assert response.status_code == 504
    assert "no pudo completar la solicitud a tiempo" in response.json()["detail"]
    assert fake_models.attempts == 3


def test_gemini_permission_error_returns_actionable_message_without_retry(monkeypatch):
    class FakeAPIError(Exception):
        def __init__(self, code, message):
            self.code = code
            self.message = message

    class FakeModels:
        def __init__(self):
            self.attempts = 0

        def generate_content(self, **kwargs):
            self.attempts += 1
            raise FakeAPIError(403, "The caller does not have permission")

    fake_models = FakeModels()

    class FakeClient:
        def __init__(self, api_key, **kwargs):
            self.models = fake_models

    monkeypatch.setenv("GEMINI_API_KEY", "test-server-key")
    monkeypatch.setenv("TMDB_API_KEY", "test-tmdb-key")
    monkeypatch.setattr(index.errors, "APIError", FakeAPIError)
    monkeypatch.setattr(index.genai, "Client", FakeClient)

    response = client.post(
        "/api/recommend",
        json={
            "content_type": "serie",
            "genres": ["misterio"],
            "preferences": "Me gustan las series de misterio.",
        },
    )

    assert response.status_code == 502
    assert "Google Gemini rechazó el acceso" in response.json()["detail"]
    assert "GEMINI_API_KEY" in response.json()["detail"]
    assert fake_models.attempts == 1


def test_gemini_timeout_returns_gateway_timeout(monkeypatch):
    class FakeModels:
        def generate_content(self, **kwargs):
            raise TimeoutException("request timed out")

    class FakeClient:
        def __init__(self, api_key, **kwargs):
            self.models = FakeModels()

    monkeypatch.setenv("GEMINI_API_KEY", "test-server-key")
    monkeypatch.setenv("TMDB_API_KEY", "test-tmdb-key")
    monkeypatch.setattr(index.genai, "Client", FakeClient)

    response = client.post(
        "/api/recommend",
        json={
            "content_type": "serie",
            "genres": ["misterio"],
            "preferences": "Me gustan las series de misterio.",
        },
    )

    assert response.status_code == 504
    assert "tardando demasiado" in response.json()["detail"]
