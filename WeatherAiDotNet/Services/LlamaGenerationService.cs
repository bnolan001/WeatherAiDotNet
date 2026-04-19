using System.Text;
using LLama;
using LLama.Common;
using LLama.Sampling;
using WeatherAiDotNet.Models;

namespace WeatherAiDotNet.Services;

internal static class LlamaGenerationService
{
    private static readonly object SyncRoot = new();
    private static LLamaWeights? s_weights;
    private static StatelessExecutor? s_executor;
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

    public static string? GetInitializationDiagnostic()
        => s_initializationDiagnostic;

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
            AntiPrompts = ["\nQuestion:", "\nUser:", "<|im_end|>"],
            SamplingPipeline = new DefaultSamplingPipeline()
        };

        var response = new StringBuilder();
        await foreach (var text in s_executor!.InferAsync(prompt, inferenceParams))
        {
            response.Append(text);
        }

        return response.Length == 0
            ? "No response returned by local model."
            : response.ToString().Trim();
    }

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
            LlamaNativeService.Configure(backend, preferGpu);

            try
            {
                var modelParams = LlamaNativeService.CreateGenerationModelParams(modelPath, gpuLayers, contextSize, threads, batchThreads, batchSize, uBatchSize);
                s_weights = LLamaWeights.LoadFromFile(modelParams);
                s_executor = new StatelessExecutor(s_weights, modelParams, logger: null);
                s_initializationDiagnostic = null;
                s_runtimeBackend = LlamaNativeService.GetRequestedRuntimeBackend(backend, preferGpu, gpuLayers);
            }
            catch (Exception gpuEx) when (preferGpu && gpuLayers > 0)
            {
                var cpuParams = LlamaNativeService.CreateGenerationModelParams(modelPath, 0, contextSize, threads, batchThreads, batchSize, uBatchSize);
                s_weights = LLamaWeights.LoadFromFile(cpuParams);
                s_executor = new StatelessExecutor(s_weights, cpuParams, logger: null);
                s_initializationDiagnostic = $"Generation model could not start on the Intel/Vulkan path and was moved to CPU-only mode. {gpuEx.Message}";
                s_runtimeBackend = "cpu";
            }

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