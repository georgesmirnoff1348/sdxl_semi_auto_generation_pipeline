import gc
import random
import time
from pathlib import Path
from PIL import Image
import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLPipeline


class CenzorStepVisualizerDPM:
    """Визуализатор промежуточных шагов (x0-эстимейтов) генерации SDXL."""

    def __init__(
        self, crooked: bool = False, lora=None, lora_weight=None, device="mps"
    ):
        model0 = "stabilityai/stable-diffusion-xl-base-1.0"
        model1 = "SG161222/RealVisXL_V4.0"

        cenzorkerneltype = (
            "СТОХАСТИЧЕСКОГО" if crooked else "ДЕТЕРМИНИРОВАННОГО"
        )
        print(f"--- ИНИЦИАЛИЗАЦИЯ {cenzorkerneltype} ЯДРА К.О.Н.Т.У.Р. ---")

        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        model_id = model1

        # Загрузка пайплайна
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            model_id, dtype=torch.float16, variant="fp16"
        )
        self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipeline.scheduler.config, use_karras_sigmas=True
        )
        self.pipeline.scheduler.algorithm_type = "dpmsolver++"

        if lora and lora_weight:
            print(f"🔗 Подключение LoRA: {lora}")
            self.pipeline.load_lora_weights(lora, adapter_name="single_lora")
            self.pipeline.set_adapters(
                ["single_lora"], adapter_weights=[lora_weight]
            )

        self.pipeline = self.pipeline.to(self.device)
        print("✅ Модель успешно загружена в ОЗУ")
        print("Используется алгоритм генерации: DPM Solver Multistep Scheduler")
        print("Используется устройство: ", self.device)
        self.pipeline.enable_attention_slicing()

    @torch.no_grad()
    def decode_latents_to_pil(self, latents: torch.Tensor) -> Image.Image:
        """Декодирует латент в полноценное изображение PIL."""
        scaling_factor = getattr(
            self.pipeline.vae.config, "scaling_factor", 0.13025
        )
        latents = latents / scaling_factor

        image_tensor = self.pipeline.vae.decode(
            latents.to(self.device, dtype=self.pipeline.dtype),
            return_dict=False,
        )[0]
        image = self.pipeline.image_processor.postprocess(
            image_tensor, output_type="pil"
        )[0]
        return image

    def generate_with_steps(
        self,
        prompt: str,
        steps_output_dir: str | Path,
        prompt_2: str = None,
        negative_prompt: str = "",
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        width: int = 1024,
        height: int = 1024,
        seed: int = None,
        output_name: str | Path = "final_output.png",
    ):
        start_gen = time.time()
        steps_dir = Path(steps_output_dir)
        steps_dir.mkdir(parents=True, exist_ok=True)

        if seed is None:
            seed = random.randint(0, 2147483647)
        print(f"🎲 Используется SEED: {seed}")
        generator = torch.Generator(device="cpu").manual_seed(seed)

        def step_callback(pipe, step_idx, timestep, callback_kwargs):
            latents = callback_kwargs["latents"]

            # Извлекаем очищенный x0-эстимейт вместо зашумленного латента
            clean_latents = latents
            scheduler = pipe.scheduler

            if (
                hasattr(scheduler, "model_outputs")
                and len(scheduler.model_outputs) > 0
            ):
                last_output = scheduler.model_outputs[-1]
                if last_output is not None:
                    step_i = (
                        scheduler.step_index
                        if scheduler.step_index is not None
                        else step_idx
                    )
                    sigma = scheduler.sigmas[step_i].to(latents.device)
                    # Вычисляем очищенное предсказание x0
                    clean_latents = latents - sigma * last_output

            # Декодируем чистый эстимейт в привычный PIL Image
            step_img = self.decode_latents_to_pil(clean_latents)
            step_path = (
                steps_dir / f"step_{step_idx + 1:02d}_t{int(timestep)}.png"
            )
            step_img.save(step_path)

            print(
                f" └─ [Кадр {step_idx + 1}/{num_inference_steps}] Сохранён x0-эстимейт: {step_path.name}"
            )
            return callback_kwargs

        out_path = Path(output_name)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        print(
            f"🚀 Запуск генерации с выгрузкой {num_inference_steps} чистых кадров x0..."
        )

        image = self.pipeline(
            prompt=prompt,
            prompt_2=prompt_2,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            width=width,
            height=height,
            callback_on_step_end=step_callback,
        ).images[0]

        image.save(out_path)
        print(
            f"✅ Итоговый результат сохранён как '{output_name}' за {time.time() - start_gen:.2f} сек."
        )

        # Очистка памяти
        if self.device == "cuda":
            torch.cuda.empty_cache()
        elif self.device == "mps":
            torch.mps.empty_cache()
        gc.collect()

        return image, seed


# ==========================================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ:
# ==========================================
if __name__ == "__main__":
    from numberedfilesaver import get_next_available_filename

    tests_dir = Path("gen/diffusortest")

    # Создаем визуализатор
    viz = CenzorStepVisualizerDPM(crooked=False)

    # Запускаем генерацию
    viz.generate_with_steps(
        prompt="half-body photo of a russian adult man wearing shirt",
        steps_output_dir=tests_dir / "frames_sequence",
        num_inference_steps=25,
        seed=456456463,
        output_name=get_next_available_filename(tests_dir, "difftest_"),
    )