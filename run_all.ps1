# Run three seeds for all four methods, one experiment at a time.
# Activate .venv before running this script.
$methods = "full_ft", "bitfit", "adapter", "lora"
$seeds = 42, 43, 44
foreach ($method in $methods) {
    foreach ($seed in $seeds) {
        python train.py --method $method --seed $seed
    }
}
python summarize.py
