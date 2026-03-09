import dotenv from "dotenv";
import { ChatOpenAI} from "@langchain/openai";
import express from "express";
import { askAgent, initAgent } from "./agent";

dotenv.config();

const PORT = 8001;
const app = express();

const model = new ChatOpenAI({
  model: "gpt-4o-mini",
  temperature: 0,
});

app.use(express.json())

app.post("/ask", async (req, res) => {
  try {
    const { question } = req.body;

    if (!question) {
      return res.status(400).json({ error: "Question is Required" });
    }

    const answer = await askAgent(question);
    return res.json({ answer });
  } catch (error) {
    console.log("Internal server error", error);
  }
});

app.listen(PORT, async () => {
  console.log(`Server is running on port ${PORT}`);
  await initAgent()
});
