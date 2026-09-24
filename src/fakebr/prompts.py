"""Conjuntos de prompts definidos em conf/prompts/<nome>.yaml."""

from __future__ import annotations

from dataclasses import dataclass

from .io import sha256


INVALID = "INVALIDA"


@dataclass(frozen=True)
class PromptSet:
    """Modelo de prompt com variantes de introdução e o vocabulário de rótulos.

    ``template`` usa os campos ``{introduction}`` e ``{text}``. ``shared_prefix_end``
    marca onde termina a parte comum a todas as notícias, usada como
    ``prompt_cache_key``; sem ele, o prefixo vai até ``{text}``.
    """

    name: str
    template: str
    introductions: dict[str, str]
    labels: dict[str, str]
    shared_prefix_end: str | None = None

    @classmethod
    def from_dict(cls, value: dict) -> "PromptSet":
        for section in ("labels", "introductions"):
            if not all(isinstance(k, str) and isinstance(v, str) for k, v in value[section].items()):
                # Ex.: "true:" sem aspas vira booleano e mudaria o nome da pasta lida.
                raise ValueError(f"Prompt {value.get('name')}: chaves e valores de {section} "
                                 "devem ser textos; use aspas em YAML (\"true\": ...)")
        prompt_set = cls(name=str(value["name"]), template=str(value["template"]),
                         introductions={str(k): str(v) for k, v in value["introductions"].items()},
                         labels={str(k): str(v) for k, v in value["labels"].items()},
                         shared_prefix_end=value.get("shared_prefix_end"))
        if "{text}" not in prompt_set.template or "{introduction}" not in prompt_set.template:
            raise ValueError(f"Prompt {prompt_set.name}: template requer {{introduction}} e {{text}}")
        if len(set(prompt_set.answers)) != len(prompt_set.answers) or INVALID in prompt_set.answers:
            raise ValueError(f"Prompt {prompt_set.name}: rótulos repetidos ou reservados")
        return prompt_set

    def to_dict(self) -> dict:
        return {"name": self.name, "template": self.template,
                "introductions": dict(self.introductions), "labels": dict(self.labels),
                "shared_prefix_end": self.shared_prefix_end}

    @property
    def answers(self) -> tuple[str, ...]:
        return tuple(self.labels.values())

    def render(self, prompt_id: str, text: str) -> str:
        if prompt_id not in self.introductions:
            raise ValueError(f"Prompt desconhecido em {self.name}: {prompt_id}")
        return self.template.format(introduction=self.introductions[prompt_id], text=text)

    def shared_prefix(self, prompt_id: str) -> str:
        if self.shared_prefix_end:
            return self.render(prompt_id, "").split(self.shared_prefix_end)[0]
        before_text = self.template.split("{text}")[0]
        return before_text.format(introduction=self.introductions[prompt_id])

    def fingerprint(self, prompt_id: str) -> str:
        return sha256(self.render(prompt_id, "{texto}"))

    def parse(self, content: str | None) -> str:
        cleaned = (content or "").strip().upper()
        for answer in self.answers:
            if cleaned == answer.upper():
                return answer
        return INVALID
