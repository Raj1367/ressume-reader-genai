import { MemoryVectorStore as i } from "langchain/vectorstores/memory";
import { Document as m } from "langchain/document";
class a extends i {
  static async fromJSON(s, c) {
    const t = JSON.parse(s);
    if (t.lc == null || t.type !== "constructor" || !Array.isArray(t.id))
      throw new Error("invalid serialization format");
    const r = new a(c);
    if (t.kwargs?.docs == null)
      throw new Error("no documents in serialization");
    {
      const e = [], n = [];
      for (const o of t.kwargs.docs)
        o.pageContent && o.vector && (e.push(new m({
          pageContent: o.pageContent,
          metadata: o.metadata || {}
        })), n.push(o.vector));
      r.addVectors(n, e);
    }
    return r;
  }
}
export {
  a as RestorableMemoryVectorStore
};
//# sourceMappingURL=RestorableMemoryVectorStore.js.map
