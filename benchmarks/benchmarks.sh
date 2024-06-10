#!/bin/sh

set -e

if [ $# -lt 1 ] || [ $0 = "-h" ] || [ $0 = "--help" ]; then
	echo "Usage: $0 tiny|small|normal|large|P_LOG_ENTRIES"
	echo "The script must be run from the project root (NOT from the benchmarks folder)"
	echo "It must also satisfy all Python requirements (use venv)"
	exit
fi

HAVI=false
P_LOG_ENTRIES=27
KEY_WIDTH=37
N_MEASUREMENTS=5
OUT="out-$1"
case $1 in
	micro)
		P_LOG_ENTRIES=22
		KEY_WIDTH=32
		N_MEASUREMENTS=1
		;;

	tiny)
		P_LOG_ENTRIES=25
		KEY_WIDTH=35
		N_MEASUREMENTS=1
		;;
	small)
		P_LOG_ENTRIES=26
		KEY_WIDTH=36
		N_MEASUREMENTS=2
		;;

	normal)
		P_LOG_ENTRIES=27
		KEY_WIDTH=37
		;;
	large)
		P_LOG_ENTRIES=28
		KEY_WIDTH=38
		;;
	manuscript)
		P_LOG_ENTRIES=27
		KEY_WIDTH=37
		N_MEASUREMENTS=10
		;;
	havi)
		HAVI=true
		;;
	*)
		P_LOG_ENTRIES=$1
		KEY_WIDTH=$(($P_LOG_ENTRIES - 10))
		;;
esac

S_LOG_ENTRIES=$((P_LOG_ENTRIES - 3))
# (Assuming little-endianness, as with all CUDA hosts)
ICEBERG_ENTRIES=$(((1 << $P_LOG_ENTRIES) + (1 << $S_LOG_ENTRIES)))
N_KEYS=$(($ICEBERG_ENTRIES * 2)) 

mkdir -p "$OUT"

if $HAVI ; then
	echo "[PROGRESS] Extracting keys..."
	xz -dc havi-log.txt.xz | ./benchmarks/txttobin.py - havi.bin
	echo "[PROGRESS] Running benchmarks..."
	./release/havi havi.bin | tee "$OUT/havi.csv"
	echo "[PROGRESS] Generating figures..."
	(
	cd "$OUT"
	../benchmarks/figures.py --kind havi havi.csv
	)
	echo "[PROGESS] Done! See the output in the directory $OUT"
	exit
fi

echo "[PROGRESS] Generating keys... (this may take a while)"
./benchmarks/generate.py -n $N_KEYS -w $KEY_WIDTH -o keys.bin
echo "[PROGRESS] Running benchmarks... (this will take a while)"
./release/rates -p $P_LOG_ENTRIES -s $S_LOG_ENTRIES -n $N_MEASUREMENTS -w $KEY_WIDTH keys.bin | tee "$OUT/rates.csv"
echo "[PROGRESS] Generating figures..."
(
cd "$OUT"
../benchmarks/figures.py rates.csv
)
echo "[PROGESS] Done! See the output in the directory $OUT"
