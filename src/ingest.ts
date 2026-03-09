import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import dotenv from "dotenv";
import fs from "fs";
import path from "path";
import { PDFParse } from "pdf-parse";
import { fileURLToPath } from "url";
import { ChromaClient, type EmbeddingFunction } from "chromadb";
import OpenAI from "openai";

dotenv.config();

const fileLocation = "/data/resume.pdf";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const openai = new OpenAI();

const res = await fetch("https://api.trychroma.com/api/v2/auth/identity", {
  headers: { "x-chroma-token": process.env.CHROMA_API_KEY! }
});
console.log("Chroma auth status:", res.status, await res.text());

export const chromaClient = new ChromaClient({
  ssl: true,
  host: "api.trychroma.com",
  port: 443,
  headers: {
    "x-chroma-token": process.env.CHROMA_API_KEY!,
  },
  tenant: process.env.CHROMA_TENANT!,
  database: process.env.CHROMA_DATABASE!,
});


const loadPdf = async (pathName: string): Promise<string> => {
  const filePath = path.join(__dirname, pathName);

  if (!fs.existsSync(filePath)) {
    throw new Error(`PDF file not found: ${filePath}`);
  }
  const rawData = fs.readFileSync(filePath);
  const parser = new PDFParse({ data: rawData });
  const pdfData = await parser.getText();

  return pdfData.text;
};

class OpenAiEmbedding implements EmbeddingFunction {
  async generate(texts: string[]): Promise<number[][]> {
    const response = await openai.embeddings.create({
      model: "text-embedding-3-small",
      input: texts,
    });
    return response.data.map((e) => e.embedding);
  }
}

export const embedPdf = async () => {
  const rawData = await loadPdf(fileLocation);

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 500,
    chunkOverlap: 100,
  });

  const chunks = await splitter.createDocuments([rawData]);

  const embedder = new OpenAiEmbedding();

  const collection = await chromaClient.getOrCreateCollection({
    name: "resume_data_collection",
    embeddingFunction: embedder,
  });

  const count = await collection.count();

  if (count === 0) {
    await collection.add({
      ids: chunks.map((_, i) => `doc_${i}`),
      documents: chunks.map((chunk) => chunk.pageContent),
      metadatas: chunks.map(() => ({ type: "resume" })),
    });

    console.log("✅ Documents inserted into collection");
  } else {
    console.log("ℹ️ Collection already contains documents");
  }

  console.log("Documents stored:", await collection.count());

  return collection; // ✅ important
};