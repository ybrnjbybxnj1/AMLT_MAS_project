import re
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser


def llm_retry():
    # retry decorator for LLM calls
    return retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((Exception,)),
        reraise=True
    )


@llm_retry()
def invoke_with_parser(llm, parser, prompt_template, **kwargs):
    # handles JSON extraction, thinking tags removal, and markdown cleanup
    if kwargs:
        try:
            prompt = prompt_template.format(**kwargs)
        except KeyError as e:
            print(f"[LLM] warning: prompt formatting failed ({e}), using raw string.")
            prompt = prompt_template
    else:
        prompt = prompt_template
    response = llm.invoke([
        SystemMessage(content=f"Respond with valid JSON matching this schema:\n{parser.get_format_instructions()}"),
        HumanMessage(content=prompt)
    ])
    content = response.content.strip()
    # remove thinking tags
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
    # remove markdown code blocks
    if content.startswith("```"):
        content = re.sub(r'^```\w*\n?|\n?```', '', content)
    # extract JSON object or array
    json_match = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', content)
    if json_match:
        content = json_match.group(1)
    # fix common escape issues
    content = re.sub(r'\\x([0-9a-fA-F]{2})', r'\\u00\1', content)
    content = content.replace(r'\x2014', '-')
    return parser.parse(content)


def clean_json_response(content: str) -> str:
    # clean LLM response to extract valid JSON
    content = content.strip()
    # remove thinking tags
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
    # remove markdown code blocks
    if content.startswith("```"):
        content = re.sub(r'^```\w*\n?', '', content)
        content = re.sub(r'\n?```$', '', content)
    # extract JSON
    json_match = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', content)
    if json_match:
        content = json_match.group(1)
    return content
