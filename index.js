// index.js

import express from 'express';
import * as dotenv from 'dotenv';
dotenv.config();
import bodyParser from 'body-parser';
import path from 'path';
import { fileURLToPath } from 'url';
import { GoogleGenerativeAIEmbeddings } from '@langchain/google-genai';
import { Pinecone } from '@pinecone-database/pinecone';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
app.use(bodyParser.json());
app.use(express.static(path.join(__dirname, 'public')));
app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'views'));

// Home route
app.get('/', (req, res) => {
    res.render('index'); // render views/index.ejs
});

// Chat route
app.post('/chat', async (req, res) => {
    try {
        const userMessage = req.body.message;

        // Initialize embeddings
        const embeddings = new GoogleGenerativeAIEmbeddings({
            apiKey: process.env.GEMINI_API_KEY,
            model: 'text-embedding-004',
        });

        // Initialize Pinecone client
        const pinecone = new Pinecone(); // reads API key & environment from .env
        const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME);

        // Convert user message to embedding vector
        const queryVector = await embeddings.embedQuery(userMessage);

        // Query Pinecone index
        const searchResults = await pineconeIndex.query({
            topK: 5,
            vector: queryVector,
            includeMetadata: true,
        });

        // Get reply from top match
        let botReply = "Sorry, I don't know the answer.";
        if (searchResults.matches && searchResults.matches.length > 0) {
            botReply = searchResults.matches[0].metadata.text || botReply;
        }

        res.json({ reply: botReply });
    } catch (err) {
        console.error(err);
        res.status(500).json({ reply: "Something went wrong!" });
    }
});

// Start server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`EduAi running on http://localhost:${PORT}`);
});
