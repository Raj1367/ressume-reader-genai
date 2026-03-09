# langchain-js-restorable-memory-vectorstore #

a restorable memory vector store for LangChainJS that works in browsers

## Overview ##

This simple class adds a `fromJSON` method to the `MemoryVectorStore` from [LangChainJS](https://github.com/langchain-ai/langchainjs) in order to be able to preserve and restore any vector store contents while still be browser-compatible (as the typical `save` and `load` methods of vector stores assume to run within a Node.js environment).

If you replace your `MemoryVectorStore` with a `RestorableMemoryVectorStore`, you can now use the original `toJSON` to serialize your vector store into a JSON object (which may then be persisted using a browser's IndexedDB or similar) and later restore it using `fromJSON`.

## Installation ##

`langchain-js-restorable-memory-vectorstore` comes as an ECMAScript module (ESM). You may either install the module using `npm` (or similar) if you plan to use a bundler:

```bash
npm install langchain-js-restorable-memory-vectorstore
```

Or you may dynamically import it using an `import` expression

```javascript
const { RestorableMemoryVectorStore } = await import "https://rozek.github.io/langchain-js-restorable-memory-vectorstore/dist/RestorableMemoryVectorStore.js"
```

## Usage in Node.js or Browser Environments ##

Assuming that you have installed the module, the `RestorableMemoryVectorStore` can be used as follows

```typescript
  import { RestorableMemoryVectorStore } from 'langchain-js-restorable-memory-vectorstore'
  import { OpenAIEmbeddings } from '@langchain/openai'
  
/**** create a vector store with documents ****/

  const Embedder = new OpenAIEmbeddings()
  const Store    = new RestorableMemoryVectorStore(Embedder)
  
/**** add some documents ****/

  await Store.addDocuments([
    { pageContent: 'Hello world', metadata: { source:'greeting' } },
    { pageContent: 'Bye world',   metadata: { source:'farewell' } },
  ])
  
/**** serialize the vector store to JSON ****/

  const Serialization = JSON.stringify(Store) // invokes "toJSON"
    
/**** later, restore the vector store ****/

  const restoredStore = await RestorableMemoryVectorStore.fromJSON(
    Serialization, Embedder
  )
      
/**** the restored store is now ready to be used ****/

  const Results = await restoredStore.similaritySearch('hello', 1)
  console.log(Results)
```

## Usage within Svelte ##

For Svelte, it is recommended to import the package in a module context:

```html
<script context="module">
  import { RestorableMemoryVectorStore } from 'langchain-js-restorable-memory-vectorstore'
  import { OpenAIEmbeddings } from '@langchain/openai'
</script>

<script>
  const Embedder = new OpenAIEmbeddings()
  const Store    = new RestorableMemoryVectorStore(Embedder)
  
/**** add some documents ****/

  await Store.addDocuments([
    { pageContent: 'Hello world', metadata: { source:'greeting' } },
    { pageContent: 'Bye world',   metadata: { source:'farewell' } },
  ])
  
/**** serialize the vector store to JSON ****/

  const Serialization = JSON.stringify(Store) // invokes "toJSON"
    
/**** later, restore the vector store ****/

  const restoredStore = await RestorableMemoryVectorStore.fromJSON(
    Serialization, Embedder
  )
      
/**** the restored store is now ready to be used ****/

  const Results = await restoredStore.similaritySearch('hello', 1)
  console.log(Results)
</script>
```

## API Reference

### `RestorableMemoryVectorStore`

extends `MemoryVectorStore` with a `fromJSON` method

#### Constructor

```typescript
new RestorableMemoryVectorStore(Embedder:EmbeddingsInterface)
```

- `Embedder`: An implementation of the LangChain `EmbeddingsInterface` to use for creating embeddings

#### Static Methods

##### `fromJSON`

```typescript
static async fromJSON(
  JSONString:string, Embedder:EmbeddingsInterface
): Promise<RestorableMemoryVectorStore>
```

restores a vector store from a serialized JSON string.

- `JSONString`: the serialized vector store in JSON format.
- `Embedder`:   an implementation of the LangChain `EmbeddingsInterface` to use for creating embeddings
- returns: a new `RestorableMemoryVectorStore` instance populated with the serialized data

## Build Instructions ##

You may easily build this package yourself.

Just install [NPM](https://docs.npmjs.com/) according to the instructions for your platform and follow these steps:

1. either clone this repository using [git](https://git-scm.com/) or [download a ZIP archive](https://github.com/rozek/langchain-js-restorable-memory-vectorstore/archive/refs/heads/main.zip) with its contents to your disk and unpack it there 
2. open a shell and navigate to the root directory of this repository
3. run `npm install` in order to install the complete build environment
4. execute `npm run build` to create a new build

You may also look into the author's [build-configuration-study](https://github.com/rozek/build-configuration-study) for a general description of his build environment.

## Test Instructions ##

`langchain-js-restorable-memory-vectorstore` comes with a few tests. Just use

```bash
npm run test
```

to run them and get a report on the console.

## License ##

[MIT License](LICENSE.md)
