data_repos="
https://archive.ics.uci.edu/ml/machine-learning-databases/covtype
"

if [[ ! -d "uci_data" ]]; then
    mkdir uci_data
    for data_url in $data_repos; do
        name=${data_url##*/}
        mkdir -p uci_data/$name
        echo $name
        wget -m -nd -r -l1 --no-parent $data_url -P ./uci_data/$name --reject "index*,INDEX*" --no-check-certificate
    done
fi
for name in uci_data/*; do
    basename=${name##*/}
    if [[ -f "$name/$basename.data.gz" ]]; then
        gzip -d "$name/$basename.data.gz"
    fi
    data_path=$name/$basename.data
    python make_datasets.py --path $data_path --mode $basename 2> /dev/null
    if [[ "$?" == 0 ]]; then
        echo "$basename processed"
    fi
done
