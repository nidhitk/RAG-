# def chunk_text(
#     text: str,
#     chunk_size: int = 50,
#     overlap: int = 10
# ):
#     words = text.split()

#     chunks = []

#     step = chunk_size - overlap

#     for start in range(0, len(words), step):

#         end = start + chunk_size

#         chunk = words[start:end]

#         if chunk:
#             chunks.append(" ".join(chunk))

#     return chunks


from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50
):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,

        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    return splitter.split_text(text)