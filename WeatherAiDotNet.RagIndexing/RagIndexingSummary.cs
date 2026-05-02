namespace WeatherAiDotNet.RagIndexing;

public sealed record RagIndexingSummary(int ReindexedDocuments, int SkippedDocuments, int TotalIndexedChunks, int TotalImageItems);
