for /l %%a in (0, 1, 9) do (
    python _main_mnist_weightGAN.py --digit --train --wSize 100 --binary_target %%a --no_mask
    rem python _main_mnist_weightGAN.py --digit --test --wSize 100 --binary_target %%a --pruning_rate 0.
)
python _main_mnist_weightGAN.py --digit --train --wSize 100 --load_model
python _main_mnist_weightGAN.py --digit --test --wSize 100 --pruning_rate 0.

for /l %%a in (0, 1, 2) do (
    python _main_mnist_weightGAN.py --balance --train --wSize 100 --binary_target %%a --no_mask
    python _main_mnist_weightGAN.py --balance --test --wSize 100 --binary_target %%a --pruning_rate 0.
)
python _main_mnist_weightGAN.py --balance --train --wSize 100 --load_model
python _main_mnist_weightGAN.py --balance --test --wSize 100 --pruning_rate 0.

goto end

for /l %%a in (0, 1, 5) do (
    python _main_mnist_weightGAN.py --digit --train --wSize 60 --no_mask
    python _main_mnist_weightGAN.py --digit --test --wSize 60 --pruning_rate 0. --no_mask
)
for /l %%a in (0, 1, 5) do (
    python _main_mnist_weightGAN.py --digit --train --wSize 30 --no_mask
    python _main_mnist_weightGAN.py --digit --test --wSize 30 --pruning_rate 0. --no_mask
)

for /l %%a in (0, 1, 2) do (
    python _main_mnist_weightGAN.py --balance --train --wSize 10 --binary_target %%a
    python _main_mnist_weightGAN.py --balance --test --wSize 10 --binary_target %%a --pruning_rate 0.
)
python _main_mnist_weightGAN.py --balance --train --wSize 10 --load_model --no_mask
python _main_mnist_weightGAN.py --balance --test --wSize 10 --pruning_rate 0. --no_mask

python _main_mnist_weightGAN.py --balance --train --wSize 10 --no_mask
python _main_mnist_weightGAN.py --balance --test --wSize 10 --pruning_rate 0. --no_mask

goto end
for /l %%a in (0, 1, 9) do (
    python _main_mnist_weightGAN.py --digit --train --wSize 60 --binary_target %%a
    python _main_mnist_weightGAN.py --digit --test --wSize 60 --binary_target %%a --pruning_rate 0.
)
python _main_mnist_weightGAN.py --digit --train --wSize 60
python _main_mnist_weightGAN.py --digit --test --wSize 60 --pruning_rate 0.

python _main_mnist_weightGAN.py --digit --train --wSize 60 --no_mask
python _main_mnist_weightGAN.py --digit --test --wSize 60 --pruning_rate 0. --no_mask


python _main_mnist_weightGAN.py --digit --train --wSize 60 --binary_target 0
python _main_mnist_weightGAN.py --digit --test --wSize 60 --binary_target 0 --pruning_rate 0.

python _main_mnist_weightGAN.py --digit --train --wSize 60 --binary_target 8
python _main_mnist_weightGAN.py --digit --test --wSize 60 --binary_target 8 --pruning_rate 0.

for /l %%a in (0, 1, 9) do (
    python _main_mnist_weightGAN.py --digit --train --wSize 32 --binary
    echo %%a
    python _main_mnist_weightGAN.py --digit --test --wSize 32 --binary
)
for /l %%a in (0, 1, 9) do (
    python _main_mnist_weightGAN.py --digit --train --wSize 60 --binary
    echo %%a
    python _main_mnist_weightGAN.py --digit --test --wSize 60 --binary
)
for /l %%a in (0, 1, 9) do (
    python _main_mnist_weightGAN.py --wine --train --wSize 4
    echo %%a
    python _main_mnist_weightGAN.py --wine --test --wSize 4
)
for /l %%a in (0, 1, 9) do (
    python _main_mnist_weightGAN.py --wine --train --wSize 6
    echo %%a
    python _main_mnist_weightGAN.py --wine --test --wSize 6
)
for /l %%a in (0, 1, 9) do (
    python _main_mnist_weightGAN.py --wine --train --wSize 10
    echo %%a
    python _main_mnist_weightGAN.py --wine --test --wSize 10
)
for /l %%a in (0, 1, 9) do (
    python _main_mnist_weightGAN.py --wine --train --wSize 20
    echo %%a
    python _main_mnist_weightGAN.py --wine --test --wSize 20
)
for /l %%a in (0, 1, 9) do (
    python _main_mnist_weightGAN.py --wine --train --wSize 30
    echo %%a
    python _main_mnist_weightGAN.py --wine --test --wSize 30
)

for /l %%a in (0, 1, 9) do (
    python _main_mnist_weightGAN.py --iris --train --wSize 4
    echo %%a
    python _main_mnist_weightGAN.py --iris --test --wSize 4
)
for /l %%a in (0, 1, 9) do (
    python _main_mnist_weightGAN.py --iris --train --wSize 5
    echo %%a
    python _main_mnist_weightGAN.py --iris --test --wSize 5
)
for /l %%a in (0, 1, 9) do (
    python _main_mnist_weightGAN.py --iris --train --wSize 10
    echo %%a
    python _main_mnist_weightGAN.py --iris --test --wSize 10
)
for /l %%a in (0, 1, 9) do (
    python _main_mnist_weightGAN.py --iris --train --wSize 20
    echo %%a
    python _main_mnist_weightGAN.py --iris --test --wSize 20
)
for /l %%a in (0, 1, 9) do (
    python _main_mnist_weightGAN.py --iris --train --wSize 30
    echo %%a
    python _main_mnist_weightGAN.py --iris --test --wSize 30
)
for /l %%a in (0, 1, 9) do (
    python _main_mnist_weightGAN.py --iris --train --wSize 40
    echo %%a
    python _main_mnist_weightGAN.py --iris --test --wSize 40
)
for /l %%a in (0, 1, 9) do (
    python _main_mnist_weightGAN.py --iris --train --wSize 50
    echo %%a
    python _main_mnist_weightGAN.py --iris --test --wSize 50
)
for /l %%a in (0, 1, 9) do (
    python _main_mnist_weightGAN.py --iris --train --wSize 60
    echo %%a
    python _main_mnist_weightGAN.py --iris --test --wSize 60
)
for /l %%a in (0, 1, 9) do (
    python _main_mnist_weightGAN.py --iris --train --wSize 70
    echo %%a
    python _main_mnist_weightGAN.py --iris --test --wSize 70
)
for /l %%a in (0, 1, 9) do (
    python _main_mnist_weightGAN.py --iris --train --wSize 80
    echo %%a
    python _main_mnist_weightGAN.py --iris --test --wSize 80
)
for /l %%a in (0, 1, 9) do (
    python _main_mnist_weightGAN.py --iris --train --wSize 90
    echo %%a
    python _main_mnist_weightGAN.py --iris --test --wSize 90
)
for /l %%a in (0, 1, 9) do (
    python _main_mnist_weightGAN.py --iris --train --wSize 100
    echo %%a
    python _main_mnist_weightGAN.py --iris --test --wSize 100
)

for /l %%a in (0, 1, 9) do (
    python _main_mnist_weightGAN.py --iris --train --wSize 2
    echo %%a
    python _main_mnist_weightGAN.py --iris --test --wSize 2
)
for /l %%a in (0, 1, 9) do (
    python _main_mnist_weightGAN.py --iris --train --wSize 3
    echo %%a
    python _main_mnist_weightGAN.py --iris --test --wSize 3
)

python _main_mnist_weightGAN.py --mnist --train --wSize 2
python _main_mnist_weightGAN.py --mnist --test --wSize 2
python _main_mnist_weightGAN.py --mnist --train --wSize 3
python _main_mnist_weightGAN.py --mnist --test --wSize 3
python _main_mnist_weightGAN.py --mnist --train --wSize 4
python _main_mnist_weightGAN.py --mnist --test --wSize 4
python _main_mnist_weightGAN.py --mnist --train --wSize 5
python _main_mnist_weightGAN.py --mnist --test --wSize 5
python _main_mnist_weightGAN.py --mnist --train --wSize 10
python _main_mnist_weightGAN.py --mnist --test --wSize 10
python _main_mnist_weightGAN.py --mnist --train --wSize 20
python _main_mnist_weightGAN.py --mnist --test --wSize 20
python _main_mnist_weightGAN.py --mnist --train --wSize 30
python _main_mnist_weightGAN.py --mnist --test --wSize 30
python _main_mnist_weightGAN.py --mnist --train --wSize 50
python _main_mnist_weightGAN.py --mnist --test --wSize 50
python _main_mnist_weightGAN.py --mnist --train --wSize 60
python _main_mnist_weightGAN.py --mnist --test --wSize 60
python _main_mnist_weightGAN.py --mnist --train --wSize 70
python _main_mnist_weightGAN.py --mnist --test --wSize 70
python _main_mnist_weightGAN.py --mnist --train --wSize 80
python _main_mnist_weightGAN.py --mnist --test --wSize 80
python _main_mnist_weightGAN.py --mnist --train --wSize 90
python _main_mnist_weightGAN.py --mnist --test --wSize 90
python _main_mnist_weightGAN.py --mnist --train --wSize 100
python _main_mnist_weightGAN.py --mnist --test --wSize 100
:end