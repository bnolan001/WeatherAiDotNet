using System.Text.Json;
using Microsoft.Data.Sqlite;
using WeatherAiDotNet.Models;

namespace WeatherAiDotNet.Services;

internal static class VectorStoreService
{
    public static void EnsureDatabase(string dbPath)
    {
        var directory = Path.GetDirectoryName(Path.GetFullPath(dbPath));
        if (!string.IsNullOrWhiteSpace(directory))
        {
            Directory.CreateDirectory(directory);
        }

        using var connection = new SqliteConnection($"Data Source={dbPath}");
        connection.Open();

        using var command = connection.CreateCommand();
        command.CommandText =
            """
            CREATE TABLE IF NOT EXISTS rag_vectors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                collection TEXT NOT NULL,
                source TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                vector_json TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_rag_vectors_collection ON rag_vectors (collection);
            CREATE INDEX IF NOT EXISTS idx_rag_vectors_source ON rag_vectors (collection, source);

            CREATE TABLE IF NOT EXISTS rag_documents (
                collection TEXT NOT NULL,
                source TEXT NOT NULL,
                fingerprint TEXT NOT NULL,
                indexed_utc TEXT NOT NULL,
                PRIMARY KEY (collection, source)
            );
            """;

        command.ExecuteNonQuery();
    }

    public static bool IsDocumentUpToDate(string dbPath, string collectionName, string source, string fingerprint)
    {
        using var connection = new SqliteConnection($"Data Source={dbPath}");
        connection.Open();

        using var command = connection.CreateCommand();
        command.CommandText =
            """
            SELECT fingerprint
            FROM rag_documents
            WHERE collection = $collection AND source = $source
            """;
        command.Parameters.AddWithValue("$collection", collectionName);
        command.Parameters.AddWithValue("$source", source);

        var existing = command.ExecuteScalar() as string;
        return string.Equals(existing, fingerprint, StringComparison.Ordinal);
    }

    public static void ReplaceDocumentVectors(
        string dbPath,
        string collectionName,
        string source,
        string fingerprint,
        IReadOnlyList<StoredVector> vectors)
    {
        using var connection = new SqliteConnection($"Data Source={dbPath}");
        connection.Open();

        using var transaction = connection.BeginTransaction();

        using (var deleteVectors = connection.CreateCommand())
        {
            deleteVectors.Transaction = transaction;
            deleteVectors.CommandText =
                """
                DELETE FROM rag_vectors
                WHERE collection = $collection AND source = $source
                """;
            deleteVectors.Parameters.AddWithValue("$collection", collectionName);
            deleteVectors.Parameters.AddWithValue("$source", source);
            deleteVectors.ExecuteNonQuery();
        }

        foreach (var item in vectors)
        {
            using var insert = connection.CreateCommand();
            insert.Transaction = transaction;
            insert.CommandText =
                """
                INSERT INTO rag_vectors (collection, source, chunk_index, text, vector_json)
                VALUES ($collection, $source, $chunkIndex, $text, $vectorJson)
                """;

            insert.Parameters.AddWithValue("$collection", item.Collection);
            insert.Parameters.AddWithValue("$source", item.Source);
            insert.Parameters.AddWithValue("$chunkIndex", item.ChunkIndex);
            insert.Parameters.AddWithValue("$text", item.Text);
            insert.Parameters.AddWithValue("$vectorJson", JsonSerializer.Serialize(item.Vector));
            insert.ExecuteNonQuery();
        }

        using (var upsertDocument = connection.CreateCommand())
        {
            upsertDocument.Transaction = transaction;
            upsertDocument.CommandText =
                """
                INSERT INTO rag_documents (collection, source, fingerprint, indexed_utc)
                VALUES ($collection, $source, $fingerprint, $indexedUtc)
                ON CONFLICT(collection, source) DO UPDATE SET
                    fingerprint = excluded.fingerprint,
                    indexed_utc = excluded.indexed_utc
                """;

            upsertDocument.Parameters.AddWithValue("$collection", collectionName);
            upsertDocument.Parameters.AddWithValue("$source", source);
            upsertDocument.Parameters.AddWithValue("$fingerprint", fingerprint);
            upsertDocument.Parameters.AddWithValue("$indexedUtc", DateTime.UtcNow.ToString("O"));
            upsertDocument.ExecuteNonQuery();
        }

        transaction.Commit();
    }

    public static bool HasAnyVectors(string dbPath, string collectionName)
    {
        using var connection = new SqliteConnection($"Data Source={dbPath}");
        connection.Open();

        using var command = connection.CreateCommand();
        command.CommandText = "SELECT COUNT(1) FROM rag_vectors WHERE collection = $collection";
        command.Parameters.AddWithValue("$collection", collectionName);

        var count = Convert.ToInt32(command.ExecuteScalar());
        return count > 0;
    }

    public static List<SearchMatch> Search(string dbPath, string collectionName, string question, float[] queryVector, int topK, int retrievalPool)
    {
        using var connection = new SqliteConnection($"Data Source={dbPath}");
        connection.Open();

        using var command = connection.CreateCommand();
        command.CommandText =
            """
            SELECT source, chunk_index, text, vector_json
            FROM rag_vectors
            WHERE collection = $collection
            """;
        command.Parameters.AddWithValue("$collection", collectionName);

        var scored = new List<SearchMatch>();
        var queryTerms = Tokenize(question).ToHashSet(StringComparer.Ordinal);

        using var reader = command.ExecuteReader();
        while (reader.Read())
        {
            var source = reader.GetString(0);
            var chunkIndex = reader.GetInt32(1);
            var text = reader.GetString(2);
            var vectorJson = reader.GetString(3);

            var vector = JsonSerializer.Deserialize<float[]>(vectorJson);
            if (vector is null || vector.Length != queryVector.Length)
            {
                continue;
            }

            var cosine = CosineSimilarity(queryVector, vector);
            var lexical = LexicalOverlapScore(queryTerms, text);
            var score = (0.8f * cosine) + (0.2f * lexical);

            scored.Add(new SearchMatch(source, chunkIndex, text, score));
        }

        return scored
            .OrderByDescending(x => x.Score)
            .Take(Math.Max(topK, retrievalPool))
            .Take(Math.Max(1, topK))
            .ToList();
    }

    private static float LexicalOverlapScore(HashSet<string> queryTerms, string text)
    {
        if (queryTerms.Count == 0)
        {
            return 0f;
        }

        var chunkTerms = Tokenize(text).ToHashSet(StringComparer.Ordinal);
        if (chunkTerms.Count == 0)
        {
            return 0f;
        }

        var intersection = queryTerms.Count(t => chunkTerms.Contains(t));
        return intersection / (float)queryTerms.Count;
    }

    private static float CosineSimilarity(float[] left, float[] right)
    {
        if (left.Length != right.Length)
        {
            return 0f;
        }

        float dot = 0;
        for (var i = 0; i < left.Length; i++)
        {
            dot += left[i] * right[i];
        }

        return dot;
    }

    private static IEnumerable<string> Tokenize(string input)
    {
        var sb = new System.Text.StringBuilder();

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
}