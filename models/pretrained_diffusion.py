from diffusers import DDPMPipeline, DDIMScheduler


def load_pretrained_ddpm_pipeline(model_id: str = "google/ddpm-cifar10-32"):
    pipe = DDPMPipeline.from_pretrained(model_id)
    return pipe


def replace_with_ddim_scheduler(pipe, num_train_timesteps: int = 1000):
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    return pipe