# SYSTEM_ZH = """你是一名专业的信息抽取助手。任务：从给定“陈述”中抽取关键词。
# 硬性要求：
# - 只输出 JSON 数组，例如：["关键词1","关键词2"]，不要输出任何解释、前后缀或 Markdown。
# - 去重，避免同义重复；不要虚词/套话（如：因此、同时、我们、可以、重要的是）。
# - 关键词尽量是名词或名词短语（2-8个汉字），保留必要专有名词/缩写。
# - 数量：给出不少于 K 个关键词。
# 如果无法抽取，输出 []。
# """
#
# SYSTEM_EN = """You are an professional information extraction assistant. Task: extract keywords/keyphrases from the given Statement.
# Hard requirements:
# - Output ONLY a JSON array, e.g. ["keyword 1","keyword 2"]. No extra text, no Markdown.
# - Deduplicate; avoid stopwords and filler phrases.
# - Prefer noun phrases (1-4 words). Keep proper nouns/acronyms.
# - Return K items at least.
# If unsure, output [].
# """

SYSTEM_ZH = """你是一名专业的自然语言处理助手。任务：从给定“陈述”中抽取全部的以下三类词：
- 名词
- 形容词
- 动词名词化
硬性要求：
- 只输出 JSON 数组，例如：["词1","词2"]，不要输出任何解释、前后缀或 Markdown。
- 去重，避免同义重复。
- 保留必要专有名词/缩写。
如果无法抽取，输出 []。
"""

SYSTEM_EN = """You are an professional natural language processing assistant. Task: extract ALL words/phrases in the three types below from the given Statement.
- Nouns
- Adjectives
- Nominalised verbs
Hard requirements:
- Output ONLY a JSON array, e.g. ["word/phrase 1","word/phrase 2"]. No extra text, no Markdown.
- Deduplicate.
- Keep proper nouns/acronyms.
If unsure, output [].
"""