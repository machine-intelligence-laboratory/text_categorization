import grpc

from ap.topic_model.v1.TopicModelBase_pb2 import DocId, Document, DocumentPack
from ap.topic_model.v1.TopicModelInference_pb2 import \
    GetDocumentsEmbeddingRequest
from ap.topic_model.v1.TopicModelInference_pb2_grpc import \
    TopicModelInferenceServiceStub

if __name__ == "__main__":
    channel = grpc.insecure_channel("localhost:50051")
    grpc_stub = TopicModelInferenceServiceStub(channel)

    docs = [
        Document(
            Id=DocId(Lo=0, Hi=0),
            Tokens=[
                "introductorio",
                "proporciona",
                "rasfondo",
                "histórico",
                "sobr",
                "seguida",
            ],
            Language="es",
        ),
        Document(
            Id=DocId(Lo=0, Hi=1),
            Tokens=[
                "bevat",
                "meer",
                "dan",
                "500",
                "analoog",
                "gestructureerde",
                "coherente",
                "ook",
            ],
            Language="nl",
        ),
    ]
    resp = grpc_stub.GetDocumentsEmbedding(
        GetDocumentsEmbeddingRequest(Pack=DocumentPack(Documents=docs))
    )

    print(resp)
