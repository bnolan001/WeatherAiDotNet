namespace WeatherAiDotNet.RagIndexing;

public record RagIndexingSummary(int ReindexedDocuments, int SkippedDocuments, int TotalIndexedChunks, int TotalImageItems);
