import random

from src.dicts_and_contants.constants import constants


def replace_with_gendered_pronouns(augment, text_col, df):
    placeholder = constants.PERSON
    assert len(set(df.Gender)) <= 3
    if augment == "single_gender":
        df = replace_with_single_option(text_col, df, placeholder)
    elif augment == "list_gender":
        df = replace_from_list(text_col, df, placeholder)
    else:
        return df

    return df


def _get_weights(df):
    # works only if M and F both less than 50%
    ratio_f = len(df[df["Gender"] == "F"]) / len(df)
    ratio_m = len(df[df["Gender"] == "M"]) / len(df)
    ratio_n = len(df[df["Gender"] == "N"]) / len(df)
    w_f = (.5 - ratio_f) / ratio_n
    w_m = (.5 - ratio_m) / ratio_n
    print("F weight", w_f, "M weight", w_m)
    return [w_f, w_m]


def replace_from_list(text_col, df, placeholder):
    # For all sentences with female indication, prepend female pronoun/ subject
    df.loc[df["Gender"] == "F", text_col] = df.loc[df["Gender"] == "F", text_col].apply(
        lambda text: text.replace(
            placeholder,
            random.choice(constants.FEMALE_LIST),
        )
    )

    print(df.loc[df["Gender"] == "F", text_col][:5])

    # For all sentences with male indication, prepend male pronoun/ subject
    df.loc[df["Gender"] == "M", text_col] = df.loc[df["Gender"] == "M", text_col].apply(
        lambda text: text.replace(
            placeholder,
            random.choice(constants.MALE_LIST),
        )
    )

    print(df.loc[df["Gender"] == "M", text_col][:5])

    random_list = random.choices([random.choice(constants.FEMALE_LIST), random.choice(constants.MALE_LIST)],
                                 weights=_get_weights(df), k=len(df[df["Gender"] == "N"]))

    for i, (index, row) in enumerate(df[df["Gender"] == "N"].iterrows()):
        df.loc[index, text_col] = row[text_col].replace(placeholder, random_list[i])

    print(df.loc[df["Gender"] == "N", text_col][:20])
    return df


def replace_with_single_option(text_col, df, placeholder):

    # For all sentences with female indication, prepend female pronoun/ subject
    df.loc[df["Gender"] == "F", text_col] = df.loc[
        df["Gender"] == "F", text_col
    ].str.replace(placeholder, constants.FEMALE_SINGLE)
    print(df.loc[df["Gender"] == "F", text_col][:5])

    # For all sentences with male indication, prepend male pronoun/ subject
    df.loc[df["Gender"] == "M", text_col] = df.loc[
        df["Gender"] == "M", text_col
    ].str.replace(placeholder, constants.MALE_SINGLE)
    print(df.loc[df["Gender"] == "M", text_col][:5])
    # For all sentences without any gender indication, gender randomly
    random_list = random.choices([constants.FEMALE_SINGLE, constants.MALE_SINGLE],
                                 weights=_get_weights(df), k=len(df[df["Gender"] == "N"]))

    for i, (index, row) in enumerate(df[df["Gender"] == "N"].iterrows()):
        df.loc[index, text_col] = row[text_col].replace(placeholder, random_list[i])

    print(df.loc[df["Gender"] == "N", text_col][:20])

    num_f = len(df[df["Gender"] == "F"]) + random_list.count(constants.FEMALE_SINGLE)
    num_m = len(df[df["Gender"] == "M"]) + random_list.count(constants.MALE_SINGLE)
    print("Counts: F", num_f, "M", num_m)
    return df


