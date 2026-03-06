import { Client } from "@opensearch-project/opensearch";
import { VectorStore } from "./base";
import { SearchFilters, VectorStoreConfig, VectorStoreResult } from "../types";

const EMBEDDING_FIELD = "vector";
const MIGRATIONS_INDEX = "memory_migrations";
const MIGRATIONS_DOC_ID = "user_id_doc";

interface OpenSearchConfig extends VectorStoreConfig {
  client?: Client;
  url?: string;
  username?: string;
  password?: string;
  collectionName: string;
  embeddingModelDims: number;
  dimension?: number;
  efSearch?: number;
  efConstruction?: number;
  m?: number;
}

export class OpenSearchDB implements VectorStore {
  private client: Client;
  private readonly indexName: string;
  private readonly dimension: number;
  private readonly efSearch: number;
  private readonly efConstruction: number;
  private readonly m: number;

  constructor(config: OpenSearchConfig) {
    if (config.client) {
      this.client = config.client;
    } else {
      const node = config.url || "http://localhost:9200";
      const clientConfig: Record<string, any> = { node };
      if (config.username && config.password) {
        clientConfig.auth = { username: config.username, password: config.password };
      }
      this.client = new Client(clientConfig);
    }

    this.indexName = config.collectionName || "mem0";
    this.dimension = config.dimension || config.embeddingModelDims || 1536;
    this.efSearch = config.efSearch ?? 512;
    this.efConstruction = config.efConstruction ?? 128;
    this.m = config.m ?? 24;

    this.initialize().catch(console.error);
  }

  async initialize(): Promise<void> {
    await this.ensureIndex(this.indexName, this.dimension);
    await this.ensureMigrationsIndex();
  }

  private async ensureIndex(name: string, dimension: number): Promise<void> {
    const exists = await this.client.indices.exists({ index: name });
    if (exists.body) return;

    try {
      await this.client.indices.create({
        index: name,
        body: {
          settings: {
            index: {
              knn: true,
              "knn.algo_param.ef_search": this.efSearch,
            },
          },
          mappings: {
            properties: {
              [EMBEDDING_FIELD]: {
                type: "knn_vector",
                dimension,
                method: {
                  name: "hnsw",
                  space_type: "cosinesimil",
                  engine: "faiss",
                  parameters: {
                    ef_construction: this.efConstruction,
                    m: this.m,
                  },
                },
              },
              payload: {
                type: "object",
                dynamic: "true" as const,
              },
            },
          },
        },
      });
    } catch (err: any) {
      // Ignore "index already exists" race condition
      if (err?.body?.error?.type !== "resource_already_exists_exception") {
        throw err;
      }
    }
  }

  private async ensureMigrationsIndex(): Promise<void> {
    const exists = await this.client.indices.exists({ index: MIGRATIONS_INDEX });
    if (exists.body) return;

    try {
      await this.client.indices.create({
        index: MIGRATIONS_INDEX,
        body: {
          mappings: {
            properties: {
              user_id: { type: "keyword" },
            },
          },
        },
      });
    } catch (err: any) {
      if (err?.body?.error?.type !== "resource_already_exists_exception") {
        throw err;
      }
    }
  }

  private buildFilterClauses(filters?: SearchFilters): Record<string, any>[] {
    if (!filters) return [];
    const clauses: Record<string, any>[] = [];
    for (const [key, value] of Object.entries(filters)) {
      if (value !== undefined && value !== null) {
        clauses.push({ term: { [`payload.${key}`]: value } });
      }
    }
    return clauses;
  }

  async insert(
    vectors: number[][],
    ids: string[],
    payloads: Record<string, any>[],
  ): Promise<void> {
    if (vectors.length === 0) return;

    const body = vectors.flatMap((vector, i) => [
      { index: { _index: this.indexName, _id: ids[i] } },
      { [EMBEDDING_FIELD]: vector, payload: payloads[i] },
    ]);

    const response = await this.client.bulk({ body, refresh: true });
    if (response.body.errors) {
      const errors = response.body.items
        .filter((item: any) => item.index?.error)
        .map((item: any) => item.index?.error);
      throw new Error(`OpenSearch bulk insert errors: ${JSON.stringify(errors)}`);
    }
  }

  async search(
    query: number[],
    limit: number = 5,
    filters?: SearchFilters,
  ): Promise<VectorStoreResult[]> {
    const filterClauses = this.buildFilterClauses(filters);

    const knnQuery: Record<string, any> = {
      [EMBEDDING_FIELD]: { vector: query, k: limit },
    };
    if (filterClauses.length > 0) {
      knnQuery[EMBEDDING_FIELD].filter = { bool: { must: filterClauses } };
    }

    const queryBody =
      filterClauses.length > 0
        ? {
            size: limit,
            query: {
              bool: {
                must: [{ knn: knnQuery }],
                filter: filterClauses,
              },
            },
          }
        : { size: limit, query: { knn: knnQuery } };

    const response = await this.client.search({ index: this.indexName, body: queryBody });
    return response.body.hits.hits.map((hit: any) => ({
      id: hit._id as string,
      payload: (hit._source?.payload ?? {}) as Record<string, any>,
      score: hit._score != null ? Number(hit._score) : undefined,
    }));
  }

  async get(vectorId: string): Promise<VectorStoreResult | null> {
    try {
      const response = await this.client.get({
        index: this.indexName,
        id: vectorId,
      });
      if (!response.body.found) return null;
      return {
        id: response.body._id,
        payload: (response.body._source?.payload ?? {}) as Record<string, any>,
      };
    } catch (err: any) {
      if (err?.statusCode === 404) return null;
      throw err;
    }
  }

  async update(
    vectorId: string,
    vector: number[],
    payload: Record<string, any>,
  ): Promise<void> {
    await this.client.index({
      index: this.indexName,
      id: vectorId,
      body: { [EMBEDDING_FIELD]: vector, payload },
      refresh: true,
    });
  }

  async delete(vectorId: string): Promise<void> {
    try {
      await this.client.delete({
        index: this.indexName,
        id: vectorId,
        refresh: true,
      });
    } catch (err: any) {
      if (err?.statusCode !== 404) throw err;
    }
  }

  async deleteCol(): Promise<void> {
    const exists = await this.client.indices.exists({ index: this.indexName });
    if (exists.body) {
      await this.client.indices.delete({ index: this.indexName });
    }
  }

  async list(
    filters?: SearchFilters,
    limit: number = 100,
  ): Promise<[VectorStoreResult[], number]> {
    const filterClauses = this.buildFilterClauses(filters);

    const baseQuery =
      filterClauses.length > 0
        ? { bool: { filter: filterClauses } }
        : { match_all: {} };

    const response = await this.client.search({
      index: this.indexName,
      body: { size: limit, query: baseQuery },
      track_total_hits: true,
    });

    const hits = response.body.hits.hits.map((hit: any) => ({
      id: hit._id as string,
      payload: (hit._source?.payload ?? {}) as Record<string, any>,
      score: hit._score != null ? Number(hit._score) : undefined,
    }));

    const total =
      typeof response.body.hits.total === "number"
        ? response.body.hits.total
        : (response.body.hits.total?.value ?? hits.length);

    return [hits, total];
  }

  async getUserId(): Promise<string> {
    try {
      const response = await this.client.get({
        index: MIGRATIONS_INDEX,
        id: MIGRATIONS_DOC_ID,
      });
      if (response.body.found && response.body._source?.user_id) {
        return response.body._source.user_id as string;
      }
    } catch (err: any) {
      if (err?.statusCode !== 404) throw err;
    }

    const randomUserId =
      Math.random().toString(36).substring(2, 15) +
      Math.random().toString(36).substring(2, 15);

    await this.client.index({
      index: MIGRATIONS_INDEX,
      id: MIGRATIONS_DOC_ID,
      body: { user_id: randomUserId },
      refresh: true,
    });

    return randomUserId;
  }

  async setUserId(userId: string): Promise<void> {
    await this.client.index({
      index: MIGRATIONS_INDEX,
      id: MIGRATIONS_DOC_ID,
      body: { user_id: userId },
      refresh: true,
    });
  }
}
