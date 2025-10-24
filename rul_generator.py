import pandas as pd
from constants import ColumnNames, FilePaths

# Kullanılacak veri dosyaları (şimdilik sadece FD001)
file_name = "train.txt"  # train_FD001.txt dosyasını bu isimle aynı klasöre koy

# Kolon isimleri
column_names = [ColumnNames.UNIT_NUMBER, ColumnNames.TIME_IN_CYCLES, 
                "op_setting_1", "op_setting_2", "op_setting_3"] \
                + [f"sensor_measurement_{i}" for i in range(1, 22)]

# Veriyi oku
df = pd.read_csv(file_name, sep=r"\s+", header=None, names=column_names)

# Her ünite için maksimum cycle
rul_df = df.groupby(ColumnNames.UNIT_NUMBER)[ColumnNames.TIME_IN_CYCLES].max().reset_index()
rul_df.columns = [ColumnNames.UNIT_NUMBER, ColumnNames.MAX_CYCLE]

# RUL hesapla
df = df.merge(rul_df, on=ColumnNames.UNIT_NUMBER, how='inner', validate='many_to_one')
df[ColumnNames.RUL] = df[ColumnNames.MAX_CYCLE] - df[ColumnNames.TIME_IN_CYCLES]
df = df.drop(ColumnNames.MAX_CYCLE, axis=1)

# CSV olarak kaydet
df.to_csv(FilePaths.TRAIN_RUL_CSV, index=False)
print(f" {FilePaths.TRAIN_RUL_CSV} dosyası oluşturuldu!")
