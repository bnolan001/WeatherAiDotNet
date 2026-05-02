using System.Text;
using LLama;
using Microsoft.Extensions.Logging;
using WeatherAiDotNet.Models;

namespace WeatherAiDotNet.Services;

/// <summary>
/// Generates floating-point embedding vectors from text using a locally loaded
/// LLamaSharp embedding model.  Vectors are used for semantic similarity search in
/// the RAG pipeline: both document chunks and the user's question are embedded so
/// the closest chunks can be retrieved as context.
/// </summary>
/// <remarks>
/// The service is statically initialised (lazy singleton) and keeps the model
/// weights loaded in memory across calls to avoid the overhead of reloading the
/// GGUF file for every chunk.  A <see cref="SyncRoot"/> lock guards initialisation
/// so the service is safe to use from a single async context.
///
/// If the embedding model cannot be loaded (missing GPU driver, unsupported hardware,
/// etc.), <see cref="GetEmbeddingAsync"/> transparently falls back to a deterministic
/// hash-based embedding.  Hash embeddings are not semantically meaningful but allow
/// the pipeline to complete and produce some results even without a working model.
/// </remarks>
public static class EmbeddingService
{
    private static readonly object SyncRoot = new();

    // Shared logger plumbing so LLamaSharp and this service emit through the same
    // Serilog-backed Microsoft.Extensions.Logging pipeline.
    private static ILogger? s_logger;

    // Lazily initialised LLamaSharp objects; null until first use.
    private static LLamaWeights? s_weights;
    private static LLamaEmbedder? s_embedder;

    // Track which configuration was used to build the current embedder so we can
    // detect when settings change and reinitialise accordingly.
    private static string? s_modelPath;
    private static string? s_backend;
    private static bool s_preferGpu;
    private static int s_gpuLayers;
    private static int s_contextSize;
    private static int s_threads;
    private static int s_batchThreads;
    private static int s_batchSize;
    private static int s_uBatchSize;
    private static string? s_initializationDiagnostic;
    private static string s_runtimeBackend = "unknown";

    /// <summary>
    /// Configures the logger factory used by this service and by the embedded
    /// LLamaSharp runtime objects it creates.
    /// </summary>
    /// <param name="loggerFactory">The shared logger factory, typically Serilog-backed.</param>
    public static void ConfigureLogging(ILoggerFactory loggerFactory)
    {
        lock (SyncRoot)
        {
            s_logger = loggerFactory.CreateLogger(typeof(EmbeddingService).FullName!);
        }
    }

    /// <summary>Returns a label identifying the active hardware backend (e.g., "vulkan" or "cpu").</summary>
    public static string GetRuntimeBackend()
        => s_runtimeBackend;

    /// <summary>
    /// Generates an embedding vector for <paramref name="input"/>.
    /// Uses the LLamaSharp model when available; falls back to a hash-based vector
    /// when <paramref name="useModelEmbeddings"/> is <see langword="false"/> or the
    /// model call fails.
    /// </summary>
    /// <param name="input">The text to embed (a chunk of PDF text or a user question).</param>
    /// <param name="fallbackEmbeddingSize">
    /// Dimension to use for the hash fallback vector.  Should match the real model's
    /// output size so the two modes produce vectors of the same length.
    /// </param>
    /// <param name="useModelEmbeddings">
    /// <see langword="true"/> if the probe confirmed a working model; otherwise the
    /// hash fallback is used directly without attempting the model.
    /// </param>
    public static async Task<float[]> GetEmbeddingAsync(
        string input,
        int fallbackEmbeddingSize,
        bool useModelEmbeddings,
        string embeddingModelPath,
        string backend,
        bool preferGpu,
        int gpuLayers,
        int contextSize,
        int threads,
        int batchThreads,
        int batchSize,
        int uBatchSize)
    {
        if (useModelEmbeddings)
        {
            var vector = await GenerateEmbeddingWithLlamaSharpAsync(
                embeddingModelPath,
                backend,
                preferGpu,
                input,
                gpuLayers,
                contextSize,
                threads,
                batchThreads,
                batchSize,
                uBatchSize);

            if (vector is { Length: > 0 })
            {
                // L2-normalise the vector so cosine similarity reduces to a dot product,
                // which is faster to compute at search time.
                Normalize(vector);
                return vector;
            }
        }

        // Model unavailable or returned an empty array; use the deterministic fallback.
        return CreateLocalEmbedding(input, fallbackEmbeddingSize);
    }

    /// <summary>
    /// Probes the embedding model with a short test string to confirm it is loadable
    /// and producing vectors.  Called once at startup before any PDF is indexed.
    /// </summary>
    /// <returns>
    /// An <see cref="EmbeddingProbeResult"/> with a non-null <c>Vector</c> on success,
    /// or a <c>Diagnostic</c> message explaining the failure.
    /// </returns>
    public static async Task<EmbeddingProbeResult> ProbeAsync(
        string embeddingModelPath,
        string backend,
        bool preferGpu,
        int gpuLayers,
        int contextSize,
        int threads,
        int batchThreads,
        int batchSize,
        int uBatchSize)
        => await TryGenerateEmbeddingWithDiagnosticsAsync(
            embeddingModelPath,
            backend,
            preferGpu,
            "embedding calibration",  // short neutral text just to exercise the model
            gpuLayers,
            contextSize,
            threads,
            batchThreads,
            batchSize,
            uBatchSize);

    /// <summary>
    /// Internal helper that routes to <see cref="TryGenerateEmbeddingWithDiagnosticsAsync"/>
    /// and discards the diagnostic wrapper, returning only the raw vector (or null).
    /// </summary>
    private static async Task<float[]?> GenerateEmbeddingWithLlamaSharpAsync(
        string embeddingModelPath,
        string backend,
        bool preferGpu,
        string input,
        int gpuLayers,
        int contextSize,
        int threads,
        int batchThreads,
        int batchSize,
        int uBatchSize)
    {
        var result = await TryGenerateEmbeddingWithDiagnosticsAsync(
            embeddingModelPath,
            backend,
            preferGpu,
            input,
            gpuLayers,
            contextSize,
            threads,
            batchThreads,
            batchSize,
            uBatchSize);

        return result.Vector;
    }

    /// <summary>
    /// Tries to generate an embedding using LLamaSharp, capturing any diagnostics
    /// (GPU fall-back notice, exception messages) alongside the result vector.
    /// This is the single place where the embedder is actually called.
    /// </summary>
    private static async Task<EmbeddingProbeResult> TryGenerateEmbeddingWithDiagnosticsAsync(
        string embeddingModelPath,
        string backend,
        bool preferGpu,
        string input,
        int gpuLayers,
        int contextSize,
        int threads,
        int batchThreads,
        int batchSize,
        int uBatchSize)
    {
        try
        {
            // Ensure the embedder is loaded with the current configuration.
            if (!TryEnsureInitialized(embeddingModelPath, backend, preferGpu, gpuLayers, contextSize, threads, batchThreads, batchSize, uBatchSize, out var diagnostic))
            {
                return new EmbeddingProbeResult(null, diagnostic);
            }

            // GetEmbeddings returns one float[] per input; we always pass a single string.
            var embeddings = await s_embedder!.GetEmbeddings(input, CancellationToken.None);
            var vector = embeddings.FirstOrDefault();

            if (vector is { Length: > 0 })
            {
                return new EmbeddingProbeResult(vector, s_initializationDiagnostic);
            }

            return new EmbeddingProbeResult(null, CombineDiagnostic(s_initializationDiagnostic, "Embedding model returned no vectors."));
        }
        catch (Exception ex)
        {
            return new EmbeddingProbeResult(null, CombineDiagnostic(s_initializationDiagnostic, ex.Message));
        }
    }

    /// <summary>
    /// Ensures the LLamaSharp embedder is initialised with the current settings.
    /// If the settings have changed since the last call the old instance is disposed
    /// and a new one is created.  Tries GPU first; falls back to CPU if GPU fails.
    /// </summary>
    /// <param name="diagnostic">
    /// Set to a warning message when the model was moved to CPU due to a GPU error,
    /// or <see langword="null"/> on clean initialisation.
    /// </param>
    /// <returns>
    /// <see langword="true"/> if the embedder is ready to use; <see langword="false"/>
    /// if both GPU and CPU initialisation failed.
    /// </returns>
    private static bool TryEnsureInitialized(
        string embeddingModelPath,
        string backend,
        bool preferGpu,
        int gpuLayers,
        int contextSize,
        int threads,
        int batchThreads,
        int batchSize,
        int uBatchSize,
        out string? diagnostic)
    {
        lock (SyncRoot)
        {
            // Return the existing instance if nothing has changed.
            if (s_embedder is not null
                && string.Equals(s_modelPath, embeddingModelPath, StringComparison.OrdinalIgnoreCase)
                && string.Equals(s_backend, backend, StringComparison.OrdinalIgnoreCase)
                && s_preferGpu == preferGpu
                && s_gpuLayers == gpuLayers
                && s_contextSize == contextSize
                && s_threads == threads
                && s_batchThreads == batchThreads
                && s_batchSize == batchSize
                && s_uBatchSize == uBatchSize)
            {
                diagnostic = s_initializationDiagnostic;
                return true;
            }

            DisposeCurrent();

            // Apply the native-library backend config (Vulkan/CPU) before loading weights.
            LlamaNativeService.Configure(backend, preferGpu);

            try
            {
                // Attempt to load the model with the requested GPU layer count.
                var modelParams = LlamaNativeService.CreateEmbeddingModelParams(embeddingModelPath, gpuLayers, contextSize, threads, batchThreads, batchSize, uBatchSize);
                s_weights = LLamaWeights.LoadFromFile(modelParams);
                s_embedder = new LLamaEmbedder(s_weights, modelParams, logger: s_logger);
                s_initializationDiagnostic = null;
                s_runtimeBackend = LlamaNativeService.GetRequestedRuntimeBackend(backend, preferGpu, gpuLayers);
                s_logger?.LogInformation("Embedding model initialized using backend {Backend}.", s_runtimeBackend);
            }
            catch (Exception gpuEx) when (preferGpu && gpuLayers > 0)
            {
                // GPU load failed; retry with 0 GPU layers to force CPU-only mode.
                try
                {
                    var cpuParams = LlamaNativeService.CreateEmbeddingModelParams(embeddingModelPath, 0, contextSize, threads, batchThreads, batchSize, uBatchSize);
                    s_weights = LLamaWeights.LoadFromFile(cpuParams);
                    s_embedder = new LLamaEmbedder(s_weights, cpuParams, logger: s_logger);
                    s_initializationDiagnostic = $"Embedding model could not start on the Intel/Vulkan path and was moved to CPU-only mode. {gpuEx.Message}";
                    s_runtimeBackend = "cpu";
                    s_logger?.LogWarning(gpuEx, "Embedding GPU initialization failed. Falling back to CPU mode.");
                }
                catch (Exception cpuEx)
                {
                    // Both GPU and CPU failed; the embedder cannot be used.
                    s_logger?.LogError(cpuEx, "Embedding model initialization failed on both GPU and CPU.");
                    diagnostic = CombineDiagnostic(gpuEx.Message, cpuEx.Message);
                    return false;
                }
            }

            // Persist the settings that produced the current embedder instance.
            s_modelPath = embeddingModelPath;
            s_backend = backend;
            s_preferGpu = preferGpu;
            s_gpuLayers = gpuLayers;
            s_contextSize = contextSize;
            s_threads = threads;
            s_batchThreads = batchThreads;
            s_batchSize = batchSize;
            s_uBatchSize = uBatchSize;
            diagnostic = s_initializationDiagnostic;
            return true;
        }
    }

    /// <summary>
    /// Disposes the current LLamaSharp objects and resets all cached state so
    /// <see cref="TryEnsureInitialized"/> will perform a fresh load on the next call.
    /// </summary>
    private static void DisposeCurrent()
    {
        s_embedder?.Dispose();
        s_weights?.Dispose();
        s_embedder = null;
        s_weights = null;
        s_modelPath = null;
        s_backend = null;
        s_preferGpu = false;
        s_gpuLayers = 0;
        s_contextSize = 0;
        s_threads = 0;
        s_batchThreads = 0;
        s_batchSize = 0;
        s_uBatchSize = 0;
        s_initializationDiagnostic = null;
        s_runtimeBackend = "unknown";
    }

    /// <summary>
    /// Merges two nullable diagnostic strings, deduplicating identical messages and
    /// separating distinct messages with a space.
    /// </summary>
    private static string? CombineDiagnostic(string? first, string? second)
    {
        var messages = new[] { first?.Trim(), second?.Trim() }
            .Where(static message => !string.IsNullOrWhiteSpace(message))
            .Distinct(StringComparer.Ordinal)
            .ToArray();

        return messages.Length == 0 ? null : string.Join(" ", messages);
    }

    /// <summary>
    /// Creates a deterministic, normalised hash-based embedding vector as a fallback
    /// when the real embedding model is unavailable.
    /// </summary>
    /// <remarks>
    /// The method tokenises the input into lowercase words, hashes each token with
    /// FNV-1a, maps the hash to a bucket index in a <paramref name="size"/>-dimensional
    /// vector, and adds ±1 (based on the hash's LSB) to that bucket.  The result is
    /// L2-normalised.  These vectors are NOT semantically meaningful — two texts about
    /// the same topic will not necessarily have a high cosine similarity — but they
    /// allow the pipeline to run end-to-end without a working model.
    /// </remarks>
    /// <param name="input">Text to hash-embed.</param>
    /// <param name="size">Number of dimensions in the output vector.</param>
    private static float[] CreateLocalEmbedding(string input, int size)
    {
        var vector = new float[size];

        foreach (var token in Tokenize(input))
        {
            var hash = StableTokenHash(token);
            var index = (int)(hash % (uint)size);

            // The least-significant bit determines the sign, giving the vector
            // both positive and negative components for better angular separation.
            var sign = (hash & 1U) == 0U ? 1f : -1f;
            vector[index] += sign;
        }

        Normalize(vector);
        return vector;
    }

    /// <summary>
    /// Splits <paramref name="input"/> into lowercase alphanumeric tokens of length ≥ 2.
    /// Short tokens (single characters) are excluded to reduce noise.
    /// </summary>
    private static IEnumerable<string> Tokenize(string input)
    {
        var sb = new StringBuilder();

        foreach (var ch in input)
        {
            if (char.IsLetterOrDigit(ch))
            {
                sb.Append(char.ToLowerInvariant(ch));
                continue;
            }

            if (sb.Length > 0)
            {
                var token = sb.ToString();
                if (token.Length >= 2)
                {
                    yield return token;
                }

                sb.Clear();
            }
        }

        if (sb.Length > 0)
        {
            var token = sb.ToString();
            if (token.Length >= 2)
            {
                yield return token;
            }
        }
    }

    /// <summary>
    /// Computes a stable 32-bit FNV-1a hash for a token string.
    /// FNV-1a is fast and has good distribution for short strings, making it
    /// suitable for the hash-embedding fallback.
    /// </summary>
    private static uint StableTokenHash(string token)
    {
        const uint offsetBasis = 2166136261;
        const uint prime = 16777619;

        var hash = offsetBasis;
        foreach (var ch in token)
        {
            hash ^= ch;
            hash *= prime;
        }

        return hash;
    }

    /// <summary>
    /// L2-normalises <paramref name="vector"/> in-place so its Euclidean length
    /// becomes 1.  After normalisation, the dot product between two vectors equals
    /// their cosine similarity, which is what <see cref="VectorStoreService"/> uses
    /// for ranking.  Vectors with zero magnitude are left unchanged to avoid division
    /// by zero.
    /// </summary>
    private static void Normalize(float[] vector)
    {
        double sum = 0;
        for (var i = 0; i < vector.Length; i++)
        {
            sum += vector[i] * vector[i];
        }

        if (sum <= 0)
        {
            return;
        }

        var norm = (float)Math.Sqrt(sum);
        for (var i = 0; i < vector.Length; i++)
        {
            vector[i] /= norm;
        }
    }
}