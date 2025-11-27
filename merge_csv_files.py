import csv

def merge_csv_files(file1, file2, output_file):
    with open(file1, 'r', newline='', encoding='utf-8') as f1, \
         open(file2, 'r', newline='', encoding='utf-8') as f2, \
         open(output_file, 'w', newline='', encoding='utf-8') as out:

        reader1 = csv.reader(f1)
        reader2 = csv.reader(f2)
        writer = csv.writer(out)

        # 读取并写入第一个文件的所有行（包括表头）
        header = next(reader1)  # 获取表头
        writer.writerow(header)
        writer.writerows(reader1)  # 写入 file1 的数据行

        # 跳过第二个文件的表头（假设和第一个一样），只写入数据行
        next(reader2, None)  # 跳过 file2 的表头
        writer.writerows(reader2)


if __name__ == "__main__":

    merge_csv_files('phishing_emails_dataset_semantic_10k.csv', 'legitimate_emails_dataset_10k.csv', 'data_set.csv')

    print(f"CSV文件已保存到: data_set.csv")
