import { chromaClient, embedPdf } from "./ingest.js";
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";

let ragChain: any;

export const initAgent = async () => {
  await embedPdf();

  const collection = await chromaClient.getCollection({
    name: "resume_data_collection",
  });

  const model = new ChatOpenAI({
    model: "gpt-4o-mini",
    temperature: 0,
  });

  const embeddings = new OpenAIEmbeddings({
    model: "text-embedding-3-small",
  });

  ragChain = async (question: string) => {
    const queryEmbedding = await embeddings.embedQuery(question);

    const result = await collection.query({
      queryEmbeddings: [queryEmbedding],
      nResults: 3,
    });

    // 🔥 Extract retrieved docs
    const docs = result.documents?.[0] || [];

    const context = docs.join("\n\n");

    const response = await model.invoke([
      {
        role: "system",
        content:
          "You are a helpful assistant that answers questions ONLY from the provided resume context.",
      },
      {
        role: "user",
        content: `Resume Context:${context}
        Question: ${question}`,
      },
    ]);

    return response.content;
  };
};

export const askAgent = async (question: string): Promise<string> => {
  if (!ragChain) {
    throw new Error("Agent not initialized. Please call initAgent()");
  }
  const res = await ragChain(question);
  return res;
};
