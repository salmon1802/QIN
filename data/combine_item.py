import pandas as pd
'''
将item_feature.parquet中的likes_level和views_level特征放在item_emb_d128之后，作为序列特征
'''


# 加载数据集
item_info = pd.read_parquet("./MicroLens_1M_x1/item_info.parquet")
item_feature = pd.read_parquet("./item_feature.parquet")

# 执行左连接，添加 likes_level 和 views_level
updated_item_info = pd.merge(
    item_info,
    item_feature[["item_id", "likes_level", "views_level"]],  # 只选择需要的列
    on="item_id",
    how="left"
)

# 可选：处理缺失值（如果需要填充默认值，取消注释以下代码）
updated_item_info["likes_level"] = updated_item_info["likes_level"].fillna(0)
updated_item_info["views_level"] = updated_item_info["views_level"].fillna(0)

# 验证结果
print("合并后的数据形状：", updated_item_info.shape)
print("前几行数据：\n", updated_item_info.head())

# 可选：保存更新后的数据集
updated_item_info.to_parquet("./MicroLens_1M_x1/item_info_add_level_feature.parquet")

print("合并完成！数据地址为：" + "./MicroLens_1M_x1/item_info_add_level_feature.parquet")
