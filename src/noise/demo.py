import noise
import filepath

def demo():
    print("введите сид")
    new_seed = int(input())
    # Генерация фрактального шума
    gen = noise.PerlinNoise(seed= new_seed)
    fractal = noise.FractalNoise(gen)
    hmap = fractal.fbm(512, 512, octaves=8, scale=0.02)

    # Применяем фильтры
    hmap = noise.NoiseFilter.power_curve(hmap, exponent=1.5)

    # Сохраняем
    noise.save_noise_as_image(hmap, filepath.OUTPUT_DIR / "heightmap.png", colormap="terrain")

    # Сэмплируем точки для деревьев
    sampler = noise.PoissonSampler(seed=123)
    pts = sampler.sample(512, 512, min_distance=20)
    noise.save_point_set(pts, filepath.OUTPUT_DIR / "trees.pts")

    print(f"Карта сохранена: heightmap.png")
    print(f"Точки сохранены: trees.pts ({len(pts.points)} точек)")

                        
if __name__ == "__main__":
    demo()