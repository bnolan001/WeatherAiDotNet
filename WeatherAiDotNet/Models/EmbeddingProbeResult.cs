namespace WeatherAiDotNet.Models;

/// <summary>
/// Captures the result of probing the embedding model at startup.
/// <see cref="Services.EmbeddingService.ProbeAsync"/> generates a short test embedding
/// to confirm the model is loadable and producing real vectors before the main
/// ingestion loop begins. If the probe fails, the app falls back to deterministic
/// hash-based embeddings so it can still run without GPU/model support.
/// </summary>
/// <param name="Vector">
/// The test embedding vector returned by the model, or <see langword="null"/> if
/// the probe failed. A non-null, non-empty vector means the model is ready.
/// </param>
/// <param name="Diagnostic">
/// Optional human-readable message describing a fallback or error condition
/// encountered during initialization (e.g., GPU load failure, CPU fallback notice).
/// </param>
public record EmbeddingProbeResult(float[]? Vector, string? Diagnostic);