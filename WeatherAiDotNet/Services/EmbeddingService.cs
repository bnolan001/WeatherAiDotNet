using System.Text;
using LLama;
using WeatherAiDotNet.Models;

namespace WeatherAiDotNet.Services;

internal static class EmbeddingService
{
    private static readonly object SyncRoot = new();
    private static LLamaWeights? s_weights;
    private static LLamaEmbedder? s_embedder;
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

    public static string GetRuntimeBackend()
        => s_runtimeBackend;

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
                Normalize(vector);
                return vector;
            }
        }

        return CreateLocalEmbedding(input, fallbackEmbeddingSize);
    }

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
            "embedding calibration",
            gpuLayers,
            contextSize,
            threads,
            batchThreads,
            batchSize,
            uBatchSize);

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
            if (!TryEnsureInitialized(embeddingModelPath, backend, preferGpu, gpuLayers, contextSize, threads, batchThreads, batchSize, uBatchSize, out var diagnostic))
            {
                return new EmbeddingProbeResult(null, diagnostic);
            }

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
            LlamaNativeService.Configure(backend, preferGpu);

            try
            {
                var modelParams = LlamaNativeService.CreateEmbeddingModelParams(embeddingModelPath, gpuLayers, contextSize, threads, batchThreads, batchSize, uBatchSize);
                s_weights = LLamaWeights.LoadFromFile(modelParams);
                s_embedder = new LLamaEmbedder(s_weights, modelParams, logger: null);
                s_initializationDiagnostic = null;
                s_runtimeBackend = LlamaNativeService.GetRequestedRuntimeBackend(backend, preferGpu, gpuLayers);
            }
            catch (Exception gpuEx) when (preferGpu && gpuLayers > 0)
            {
                try
                {
                    var cpuParams = LlamaNativeService.CreateEmbeddingModelParams(embeddingModelPath, 0, contextSize, threads, batchThreads, batchSize, uBatchSize);
                    s_weights = LLamaWeights.LoadFromFile(cpuParams);
                    s_embedder = new LLamaEmbedder(s_weights, cpuParams, logger: null);
                    s_initializationDiagnostic = $"Embedding model could not start on the Intel/Vulkan path and was moved to CPU-only mode. {gpuEx.Message}";
                    s_runtimeBackend = "cpu";
                }
                catch (Exception cpuEx)
                {
                    diagnostic = CombineDiagnostic(gpuEx.Message, cpuEx.Message);
                    return false;
                }
            }

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

    private static string? CombineDiagnostic(string? first, string? second)
    {
        var messages = new[] { first?.Trim(), second?.Trim() }
            .Where(static message => !string.IsNullOrWhiteSpace(message))
            .Distinct(StringComparer.Ordinal)
            .ToArray();

        return messages.Length == 0 ? null : string.Join(" ", messages);
    }

    private static float[] CreateLocalEmbedding(string input, int size)
    {
        var vector = new float[size];

        foreach (var token in Tokenize(input))
        {
            var hash = StableTokenHash(token);
            var index = (int)(hash % (uint)size);
            var sign = (hash & 1U) == 0U ? 1f : -1f;
            vector[index] += sign;
        }

        Normalize(vector);
        return vector;
    }

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