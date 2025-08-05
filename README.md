# WTNet: A Weather Transfer Network for Domain-Adaptive All-In-One Adverse Weather Image Restoration (BMVC 2025)

Si-Yu Huang, Fu-Jen Tsai, Chia-Wen Lin, Yen-Yu Lin

>Abstract: All-in-one adverse weather image restoration has attracted increasing attention due to its potential to recover high-quality images with a single model.
However, existing methods often exhibit significant performance drops due to the domain gap between training and testing weather conditions. 
Moreover, they typically achieve only average, rather than optimal, performance across different weather conditions, when compared to weather-specific approaches.
To address these two issues, we propose a novel Weather Transfer Network (WTNet), which fine-tunes all-in-one models to enhance their performance during testing.
Recognizing the unavailability of paired degraded-clean images at test time, WTNet transfers degradation patterns from the testing images in an unseen target domain to clean images in the source domain, thereby generating the fine-tuning sets for enabling domain adaptation. 
Additionally, by leveraging the fine-tuning sets, all-in-one models can be dynamically adapted to weather-specific or mixed weather models based on the transferred degradation patterns observed during testing.
Experimental results demonstrate that WTNet can significantly enhance state-of-the-art all-in-one models across real-world image deraining, desnowing, and dehazing benchmarks.
