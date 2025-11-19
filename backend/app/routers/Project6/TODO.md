add a trimmer to reduce file size length of .ndjson files
add a custom resolution option instead of just 28x28

Label Smoothing (MEDIUM IMPACT)
Change real labels from 1.0 → 0.9, fake from 0.0 → 0.1
Prevents discriminator from getting overconfident
Easy to implement, helps convergence