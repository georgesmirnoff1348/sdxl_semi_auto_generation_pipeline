from dataclasses import dataclass, field
from typing import Optional, Any, Dict
from collections.abc import Mapping
import torch


@dataclass
class FactorPrompts(Mapping):
    prompt: str = ""
    negative_prompt: str = ""
    # Для SDXL: второй текстовый энкодер или промпт для рефайнера
    prompt_2: Optional[str] = None
    negative_prompt_2: Optional[str] = None
    
    @classmethod
    def soviet_citizen(
        cls, 
        composition: str = "half-body photo",
        age: str = "adult",
        nationality: str = "russian",
        gender: str = "character", 
        clothing: str = "casual clothes", 
    ) -> "FactorPrompts":
        """Процедурный конструктор текстового описания человека."""
        prompt_healthy = (
            f"A {composition} of a {age} {nationality} {gender}, "
            f"wearing {clothing}, distinct ethnic facial features, authentic eyes, "
            f"realistic skin texture, solid neutral studio background, 35mm photograph"
        )

        prompt2_healthy = (
            f"A {composition} of a {age} {nationality} {gender}, wearing {clothing}, "
            f"1980s soviet casual photo, "
            f"soft studio lighting, sharp focus, analogue film grain"
        )

        negative_healthy = (
            "distorted face, extreme close-up, macro shot, cropped head, military uniform, "
            "siloviki, 3d render, anime, smooth plastic skin, blurry, crooked, "
            "digital artifacts, illustration, drawing, painting, unrealistic, cartoon, "
            "comic, deformed, poster, cars, flags, text"
        )
        return cls(
            prompt=prompt_healthy,
            prompt_2 = prompt2_healthy,
            negative_prompt=negative_healthy,
            negative_prompt_2= ""
        )
    @classmethod
    def spore_syndrome(
        cls,
        nationality: str = "soviet",
        gender: str = "woman"
    ):
        """
        Хороший промпт генерации споровых мутантов
        """
        prompt_sporous = (f"photograph of a mutated {nationality} {gender} with gigantic, unnaturally bulging eyes, "
            f"exophthalmic, heavy wrinkles, grey skin, cracked and dry skin, "
            f"grotesque and eerie appearance, cracked stone-like skin texture with mold, concrete background, "
            f"1980s aesthetic, photorealistic, vintage studio photo, film grain, muted colors")

        negative_prompt = ("hat, green, smooth skin, beauty, clean, makeup, 3d render, anime, plastic, skeleton, "
          "blurry, illustration, drawing, cartoon, green, frame, torn corners, crumpled edges")
        return cls(
            prompt = prompt_sporous,
            negative_prompt = negative_prompt,
            prompt_2 = "",
            negative_prompt_2 = ""
        )
    def _as_dict(self) -> Dict[str, Any]:
            d = {
                "prompt": self.prompt,
                "prompt_2": self.prompt_2,
                "negative_prompt": self.negative_prompt,
                "negative_prompt_2": self.negative_prompt_2,
            }
            return d

    def __getitem__(self, key):
        return self._as_dict()[key]

    def __iter__(self):
        return iter(self._as_dict())

    def __len__(self):
        return len(self._as_dict())


@dataclass
class FactorInferenceParameters(Mapping):
    num_inference_steps: int = 20
    guidance_scale: float = 7.0
    seed: Optional[int] = None
    height: int = 1024
    width: int = 1024
    extra: Dict[str, Any] = field(default_factory=dict)

    def _as_dict(self) -> Dict[str, Any]:
        d = {
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "height": self.height,
            "width": self.width,
            "seed": self.seed
        }
        d.update(self.extra)
        return d

    def __getitem__(self, key):
        return self._as_dict()[key]

    def __iter__(self):
        return iter(self._as_dict())

    def __len__(self):
        return len(self._as_dict())


# Usage:
# pipe(**prompts, **params)
# where `prompts` is an instance of FactorPrompts
# and `params` is an instance of FactorInferenceParameters