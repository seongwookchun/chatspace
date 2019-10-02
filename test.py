from konlpy.tag import Mecab
from chatspace import ChatSpace

# Mecab으로 띄어쓰기 처리가 된 문장을 넣어야합니다.

tokenizer = Mecab()
segmentation = ChatSpace()

sent = "아야 난 놀고 싶다! 넌 회사에 언제 올래 "
tokenized_sent = " ".join(tokenizer.morphs(sent))

print(segmentation.space(tokenized_sent))