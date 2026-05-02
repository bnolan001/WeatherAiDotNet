namespace WeatherAiDotNet.RagIndexing;

public sealed record RagStoredVector(int ChunkIndex, string Text, float[] Vector);
