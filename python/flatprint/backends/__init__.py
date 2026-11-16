"""Backend registry. Imports are lazy, so torch/httpx load only when used."""


def get_backend(name: str, model: str, options: dict):
    if name == "diffusers":
        from flatprint.backends.diffusers_local import DiffusersBackend
        return DiffusersBackend(model, options)
    if name == "vae-roundtrip":
        from flatprint.backends.diffusers_local import VaeRoundtripBackend
        return VaeRoundtripBackend(model, options)
    if name == "openai":
        from flatprint.backends.openai_images import OpenAIImagesBackend
        return OpenAIImagesBackend(model, options)
    if name == "gemini":
        from flatprint.backends.gemini_images import GeminiImagesBackend
        return GeminiImagesBackend(model, options)
    raise ValueError(f"unknown backend '{name}'")
