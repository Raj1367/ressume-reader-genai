import { MemoryVectorStore } from 'langchain/vectorstores/memory';
import { Embeddings } from '@langchain/core/embeddings';
/*******************************************************************************
*                                                                              *
*                         RestorableMemoryVectorStore                          *
*                                                                              *
*******************************************************************************/
export type Serialized = {
    lc: number;
    type: 'constructor';
    id: string[];
    kwargs: Record<string, any>;
};
/**** RestorableMemoryVectorStore ****/
export declare class RestorableMemoryVectorStore extends MemoryVectorStore {
    static fromJSON(JSONString: string, Embedder: Embeddings): Promise<RestorableMemoryVectorStore>;
}
