using WeatherAiDotNet.Configuration;
using WeatherAiDotNet.Models;
using WeatherAiDotNet.Services;

namespace WeatherAiDotNet.Tests;

/// <summary>
/// Integration tests for the RAG (Retrieval-Augmented Generation) system.
/// These tests verify that:
/// 1. The vector database can be populated with PDF content
/// 2. Semantic search retrieves relevant chunks
/// 3. The generation model produces answers containing expected information
/// </summary>
public class RagIntegrationTests : IAsyncLifetime
{
    private readonly string _testDbPath;
    private readonly string _testCollectionName = "test-rag-afh11-203v2";
    private const string Afh11203v2PdfName = "afh11-203v2.pdf";

    private AppOptions _appOptions = null!;

    public RagIntegrationTests()
    {
        _testDbPath = Path.Combine(Path.GetTempPath(), $"rag-test-{Guid.NewGuid():N}.db");
    }

    /// <summary>
    /// Initialize the RAG system with AFH11-203v2 PDF.
    /// This runs once before any tests execute.
    /// </summary>
    public async Task InitializeAsync()
    {
        var pdfPath = FindPdfFile(Afh11203v2PdfName);
        Assert.True(File.Exists(pdfPath), $"PDF not found: {pdfPath}");

        var appDirectory = AppContext.BaseDirectory;
        var modelPath = Path.Combine(appDirectory, "AiModels", "qwen2.5-coder-7b-instruct-q4_k_m.gguf");
        var embeddingModelPath = Path.Combine(appDirectory, "AiModels", "bge-base-en-v1.5-q4_k_m.gguf");

        Assert.True(File.Exists(modelPath), $"Generation model not found: {modelPath}");
        Assert.True(File.Exists(embeddingModelPath), $"Embedding model not found: {embeddingModelPath}");

        _appOptions = new AppOptions
        {
            PdfFolderPath = Path.GetDirectoryName(pdfPath)!,
            ModelPath = modelPath,
            EmbeddingModelPath = embeddingModelPath,
            LlamaBackend = "vulkan",
            PreferGpu = true,
            DbPath = _testDbPath,
            CollectionName = _testCollectionName,
            TopK = 6,
            RetrievalPool = 24,
            EmbeddingSize = 384,
            GpuLayers = 999,
            ContextSize = 4096,
            Threads = Environment.ProcessorCount,
            BatchThreads = Environment.ProcessorCount,
            BatchSize = 512,
            UBatchSize = 256,
            IncludeImages = false,
            ImagesOutputPath = Path.Combine(Path.GetTempPath(), "test-extracted-images"),
            OcrCliPath = string.Empty
        };

        // Initialize the vector database
        VectorStoreService.EnsureDatabase(_appOptions.DbPath);

        // Index the PDF
        await IndexPdfAsync(pdfPath);
    }

    /// <summary>
    /// Clean up temporary test resources.
    /// </summary>
    public Task DisposeAsync()
    {
        try
        {
            if (File.Exists(_testDbPath))
            {
                File.Delete(_testDbPath);
            }
        }
        catch
        {
            // Ignore cleanup errors
        }

        return Task.CompletedTask;
    }

    /// <summary>
    /// Test that semantic search retrieves relevant content for a question.
    /// This verifies the embedding and retrieval pipeline without requiring answer generation.
    /// </summary>
    [Fact]
    public async Task SemanticSearch_RetrievesRelevantChunks()
    {
        var question = "What is a synoptic observation?";

        var questionVector = await EmbeddingService.GetEmbeddingAsync(
            question,
            _appOptions.EmbeddingSize,
            true,
            _appOptions.EmbeddingModelPath,
            _appOptions.LlamaBackend,
            _appOptions.PreferGpu,
            _appOptions.GpuLayers,
            _appOptions.ContextSize,
            _appOptions.Threads,
            _appOptions.BatchThreads,
            _appOptions.BatchSize,
            _appOptions.UBatchSize);

        var matches = VectorStoreService.Search(
            _appOptions.DbPath,
            _appOptions.CollectionName,
            question,
            questionVector,
            _appOptions.TopK,
            _appOptions.RetrievalPool);

        Assert.NotEmpty(matches);
        var matchText = string.Join(" ", matches.Select(m => m.Text.ToLowerInvariant()));
        Assert.Contains("synoptic", matchText);
    }

    /// <summary>
    /// Test that the RAG system can answer questions about AFH11-203v2 content.
    /// Verifies that generated answers contain expected keywords from the document.
    /// </summary>
    [Theory]
    [MemberData(nameof(GetTestCases))]
    public async Task RagSystem_AnswersQuestionAboutDocument(RagTestCase testCase)
    {
        var questionVector = await EmbeddingService.GetEmbeddingAsync(
            testCase.Question,
            _appOptions.EmbeddingSize,
            true,
            _appOptions.EmbeddingModelPath,
            _appOptions.LlamaBackend,
            _appOptions.PreferGpu,
            _appOptions.GpuLayers,
            _appOptions.ContextSize,
            _appOptions.Threads,
            _appOptions.BatchThreads,
            _appOptions.BatchSize,
            _appOptions.UBatchSize);

        var matches = VectorStoreService.Search(
            _appOptions.DbPath,
            _appOptions.CollectionName,
            testCase.Question,
            questionVector,
            _appOptions.TopK,
            _appOptions.RetrievalPool);

        Assert.NotEmpty(matches);//, $"No relevant context found for: {testCase.Question}");

        var context = string.Join("\n\n---\n\n", matches.Select((m, idx) => $"[{idx + 1}] {m.Text}"));
        var prompt = $"""
        Answer the following question based on the provided context. Be concise and accurate.

        Context:
        {context}

        Question: {testCase.Question}

        Answer:
        """;

        var answer = await LlamaGenerationService.GenerateAnswerAsync(
            _appOptions.ModelPath,
            prompt,
            maxTokens: 200,
            _appOptions.LlamaBackend,
            _appOptions.PreferGpu,
            _appOptions.GpuLayers,
            _appOptions.ContextSize,
            _appOptions.Threads,
            _appOptions.BatchThreads,
            _appOptions.BatchSize,
            _appOptions.UBatchSize);

        var answerLower = answer.ToLowerInvariant();
        var foundKeywords = testCase.ExpectedKeywords
            .Where(k => answerLower.Contains(k.ToLowerInvariant()))
            .ToList();

        Assert.NotEmpty(foundKeywords);/*, 
            $"Answer did not contain expected keywords for: {testCase.Question}\n" +
            $"Expected any of: {string.Join(", ", testCase.ExpectedKeywords)}\n" +
            $"Got: {answer}");*/
    }

    /// <summary>
    /// Test that retrieved chunks are relevant to the question.
    /// This validates the vector embedding and similarity search without generation.
    /// </summary>
    [Fact]
    public async Task VectorRetrieval_MatchesQuestionIntent()
    {
        var question = "How is wind speed measured?";

        var questionVector = await EmbeddingService.GetEmbeddingAsync(
            question,
            _appOptions.EmbeddingSize,
            true,
            _appOptions.EmbeddingModelPath,
            _appOptions.LlamaBackend,
            _appOptions.PreferGpu,
            _appOptions.GpuLayers,
            _appOptions.ContextSize,
            _appOptions.Threads,
            _appOptions.BatchThreads,
            _appOptions.BatchSize,
            _appOptions.UBatchSize);

        var matches = VectorStoreService.Search(
            _appOptions.DbPath,
            _appOptions.CollectionName,
            question,
            questionVector,
            topK: 3,
            _appOptions.RetrievalPool);

        Assert.NotEmpty(matches);
        var topMatch = matches[0];
        var topMatchLower = topMatch.Text.ToLowerInvariant();

        Assert.True(
            topMatchLower.Contains("wind") || topMatchLower.Contains("speed") || topMatchLower.Contains("knot"),
            $"Top match not relevant to wind speed: {topMatch.Text}");
    }

    /// <summary>
    /// Test that the database correctly stores and tracks documents.
    /// </summary>
    [Fact]
    public void VectorDatabase_TracksPdfMetadata()
    {
        var hasVectors = VectorStoreService.HasAnyVectors(_appOptions.DbPath, _appOptions.CollectionName);
        Assert.True(hasVectors, "No vectors found in database after indexing");
    }

    public static IEnumerable<object[]> GetTestCases()
        => RagTestDatasets.Afh11203v2TestCases.Select(tc => new object[] { tc });

    private async Task IndexPdfAsync(string pdfPath)
    {
        var relativePdfPath = Path.GetFileName(pdfPath);
        var fingerprint = BuildDocumentFingerprint(pdfPath);

        // Skip if already indexed
        if (VectorStoreService.IsDocumentUpToDate(_appOptions.DbPath, _appOptions.CollectionName, relativePdfPath, fingerprint))
        {
            return;
        }

        string pdfText;
        try
        {
            pdfText = PdfContentService.ReadPdfText(pdfPath);
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"Failed to read PDF: {pdfPath}", ex);
        }

        if (string.IsNullOrWhiteSpace(pdfText))
        {
            throw new InvalidOperationException($"PDF contains no readable text: {pdfPath}");
        }

        var chunkIndex = 0;
        var fileVectors = new List<StoredVector>();

        var chunks = PdfContentService.ChunkText(pdfText, chunkSize: 1200, overlap: 200).ToList();
        foreach (var chunk in chunks)
        {
            var vector = await EmbeddingService.GetEmbeddingAsync(
                chunk,
                _appOptions.EmbeddingSize,
                true,
                _appOptions.EmbeddingModelPath,
                _appOptions.LlamaBackend,
                _appOptions.PreferGpu,
                _appOptions.GpuLayers,
                _appOptions.ContextSize,
                _appOptions.Threads,
                _appOptions.BatchThreads,
                _appOptions.BatchSize,
                _appOptions.UBatchSize);

            fileVectors.Add(new StoredVector(_appOptions.CollectionName, relativePdfPath, chunkIndex++, chunk, vector));
        }

        VectorStoreService.ReplaceDocumentVectors(
            _appOptions.DbPath,
            _appOptions.CollectionName,
            relativePdfPath,
            fingerprint,
            fileVectors);
    }

    private static string FindPdfFile(string pdfName)
    {
        // Check the typical data directory relative to the main app
        var appDirectory = AppContext.BaseDirectory;
        var mainProjectDataPath = Path.Combine(appDirectory, "..", "..", "WeatherAiDotNet", "Data", pdfName);
        if (File.Exists(mainProjectDataPath))
        {
            return Path.GetFullPath(mainProjectDataPath);
        }

        // Check relative to current test directory
        var testDataPath = Path.Combine(AppContext.BaseDirectory, "Data", pdfName);
        if (File.Exists(testDataPath))
        {
            return testDataPath;
        }

        // Fallback to main app output directory
        var fallbackPath = Path.Combine(appDirectory, "Data", pdfName);
        return fallbackPath;
    }

    private static string BuildDocumentFingerprint(string filePath)
    {
        var fileInfo = new FileInfo(filePath);
        return $"{fileInfo.Length}:{fileInfo.LastWriteTimeUtc.Ticks}";
    }
}
