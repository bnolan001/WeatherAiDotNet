using System.Text;
using LLama;
using LLama.Common;
using LLama.Sampling;
using WeatherAiDotNet.Models;

namespace WeatherAiDotNet.Services;

/// <summary>
/// Runs text-generation inference against a locally loaded LLamaSharp model.
/// Takes a fully-constructed RAG prompt (question + retrieved context) and streams
/// the model's answer tokens into a single response string.
/// </summary>
/// <remarks>
/// Like <see cref="EmbeddingService"/>, this service keeps the model weights in
/// memory via a static lazy singleton to avoid reloading the GGUF file on every
/// question.  A <see cref="SyncRoot"/> lock guards the one-time initialisation.
/// If the model cannot be loaded on the GPU path, the service automatically retries
/// with CPU-only settings before giving up.
/// </remarks>
public static class LlamaGenerationService
{
    private static readonly object SyncRoot = new();

    // Lazily initialised LLamaSharp objects; null until first call.
    private static LLamaWeights? s_weights;
    private static StatelessExecutor? s_executor;

    // Track the configuration used to build the current executor so we can detect
    // settings changes and reinitialise if necessary.
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

    /// <summary>Returns a label identifying the active hardware backend (e.g., "vulkan" or "cpu").</summary>
    public static string GetRuntimeBackend()
        => s_runtimeBackend;

    /// <summary>Returns any diagnostic message recorded during model initialisation, or <see langword="null"/>.</summary>
    public static string? GetInitializationDiagnostic()
        => s_initializationDiagnostic;

    /// <summary>
    /// Probes the generation model by attempting to initialise it.  Called once at
    /// startup to confirm the model is loadable before the ingestion loop begins.
    /// </summary>
    /// <returns>
    /// A <see cref="ToolCheckResult"/> with <c>Success = true</c> when the model
    /// loaded without throwing, or <c>Success = false</c> with the exception message.
    /// </returns>
    public static Task<ToolCheckResult> ProbeAsync(
        string modelPath,
        string backend,
        bool preferGpu,
        int gpuLayers,
        int contextSize,
        int threads,
        int batchThreads,
        int batchSize,
        int uBatchSize)
    {
        try
        {
            EnsureInitialized(modelPath, backend, preferGpu, gpuLayers, contextSize, threads, batchThreads, batchSize, uBatchSize);
            return Task.FromResult(new ToolCheckResult(true, s_initializationDiagnostic ?? string.Empty));
        }
        catch (Exception ex)
        {
            return Task.FromResult(new ToolCheckResult(false, ex.Message));
        }
    }

    /// <summary>
    /// Sends <paramref name="prompt"/> to the local generation model and returns the
    /// complete response as a single trimmed string.
    /// </summary>
    /// <param name="modelPath">Path to the generation model GGUF file.</param>
    /// <param name="prompt">The fully-assembled RAG prompt (context + question).</param>
    /// <param name="maxTokens">
    /// Maximum number of tokens the model may generate.  Keeps responses concise and
    /// prevents the model from running forever on open-ended questions.
    /// </param>
    /// <returns>
    /// The generated answer string, or a fallback message if the model returned nothing.
    /// </returns>
    public static async Task<string> GenerateAnswerAsync(
        string modelPath,
        string prompt,
        int maxTokens,
        string backend,
        bool preferGpu,
        int gpuLayers,
        int contextSize,
        int threads,
        int batchThreads,
        int batchSize,
        int uBatchSize)
    {
        EnsureInitialized(modelPath, backend, preferGpu, gpuLayers, contextSize, threads, batchThreads, batchSize, uBatchSize);

        var inferenceParams = new InferenceParams
        {
            MaxTokens = maxTokens,

            // Stop generating when these sequences appear so the model doesn't bleed
            // into the next "turn" of a chat format that was not used here.
            AntiPrompts = ["\nQuestion:", "\nUser:", "<|im_end|>"],

            SamplingPipeline = new DefaultSamplingPipeline()
        };

        // StatelessExecutor streams tokens as they are generated; we collect them all.
        var response = new StringBuilder();
        await foreach (var text in s_executor!.InferAsync(prompt, inferenceParams))
        {
            response.Append(text);
        }

        return response.Length == 0
            ? "No response returned by local model."
            : response.ToString().Trim();
    }

    /// <summary>
    /// Initialises the LLamaSharp executor with the supplied settings, reusing the
    /// existing instance when the settings match.  Tries GPU first and falls back to
    /// CPU-only on failure.
    /// </summary>
    private static void EnsureInitialized(
        string modelPath,
        string backend,
        bool preferGpu,
        int gpuLayers,
        int contextSize,
        int threads,
        int batchThreads,
        int batchSize,
        int uBatchSize)
    {
        lock (SyncRoot)
        {
            // Return early if the executor is already loaded with the same settings.
            if (s_executor is not null
                && string.Equals(s_modelPath, modelPath, StringComparison.OrdinalIgnoreCase)
                && string.Equals(s_backend, backend, StringComparison.OrdinalIgnoreCase)
                && s_preferGpu == preferGpu
                && s_gpuLayers == gpuLayers
                && s_contextSize == contextSize
                && s_threads == threads
                && s_batchThreads == batchThreads
                && s_batchSize == batchSize
                && s_uBatchSize == uBatchSize)
            {
                return;
            }

            DisposeCurrent();

            // Configure the native backend (Vulkan/CPU) before loading any weights.
            LlamaNativeService.Configure(backend, preferGpu);

            try
            {
                // Attempt to load with the requested GPU layer count.
                var modelParams = LlamaNativeService.CreateGenerationModelParams(modelPath, gpuLayers, contextSize, threads, batchThreads, batchSize, uBatchSize);
                s_weights = LLamaWeights.LoadFromFile(modelParams);
                s_executor = new StatelessExecutor(s_weights, modelParams, logger: null);
                s_initializationDiagnostic = null;
                s_runtimeBackend = LlamaNativeService.GetRequestedRuntimeBackend(backend, preferGpu, gpuLayers);
            }
            catch (Exception gpuEx) when (preferGpu && gpuLayers > 0)
            {
                // GPU load failed; retry in CPU-only mode so the app can still answer questions.
                var cpuParams = LlamaNativeService.CreateGenerationModelParams(modelPath, 0, contextSize, threads, batchThreads, batchSize, uBatchSize);
                s_weights = LLamaWeights.LoadFromFile(cpuParams);
                s_executor = new StatelessExecutor(s_weights, cpuParams, logger: null);
                s_initializationDiagnostic = $"Generation model could not start on the Intel/Vulkan path and was moved to CPU-only mode. {gpuEx.Message}";
                s_runtimeBackend = "cpu";
            }

            // Persist the settings so we can detect changes on the next call.
            s_modelPath = modelPath;
            s_backend = backend;
            s_preferGpu = preferGpu;
            s_gpuLayers = gpuLayers;
            s_contextSize = contextSize;
            s_threads = threads;
            s_batchThreads = batchThreads;
            s_batchSize = batchSize;
            s_uBatchSize = uBatchSize;
        }
    }

    /// <summary>
    /// Disposes the current LLamaSharp objects and resets all cached state so the
    /// next call to <see cref="EnsureInitialized"/> performs a fresh load.
    /// </summary>
    private static void DisposeCurrent()
    {
        s_executor?.Context.Dispose();
        s_weights?.Dispose();
        s_executor = null;
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
}