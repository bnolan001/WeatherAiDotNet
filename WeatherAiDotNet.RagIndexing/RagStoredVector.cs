namespace WeatherAiDotNet.RagIndexing;

public record RagStoredVector(int ChunkIndex, string Text, float[] Vector);
