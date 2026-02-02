# Translation Pretraining 

Please refer to the general pretraining [instructions](../../README.md). 


We use a dataset called[ sentence-transformers/parallel-sentences-ccmatrix](https://huggingface.co/datasets/sentence-transformers/parallel-sentences-ccmatrix) which contains 
translation pairs between English and other languages. So, we support translating English to other languages and non-English languages to English. 

## data.py
The following languages are supported with the following number of cases. 

| id  | language        | number of cases |
| --- | --------------- | --------------: |
| af  | Afrikaans       |       8,694,461 |
| ar  | Arabic          |      49,697,322 |
| ast | Asturian        |       2,956,618 |
| az  | Azerbaijani     |       1,251,254 |
| be  | Belarusian      |       1,885,446 |
| bg  | Bulgarian       |      44,635,282 |
| bn  | Bengali         |      10,074,620 |
| br  | Breton          |         454,175 |
| ca  | Catalan         |      21,284,430 |
| ceb | Cebuano         |         962,549 |
| cs  | Czech           |      56,307,029 |
| da  | Danish          |      52,273,664 |
| de  | German          |     247,470,736 |
| el  | Greek           |      49,262,631 |
| eo  | Esperanto       |      15,418,393 |
| es  | Spanish         |     409,061,333 |
| et  | Estonian        |      22,007,049 |
| eu  | Basque          |       7,778,871 |
| fa  | Persian         |      24,597,533 |
| fi  | Finnish         |      35,982,562 |
| fr  | French          |     328,595,738 |
| fy  | Western Frisian |       1,372,321 |
| ga  | Irish           |       1,076,420 |
| gd  | Scottish Gaelic |         310,351 |
| gl  | Galician        |      13,178,507 |
| ha  | Hausa           |       5,861,080 |
| he  | Hebrew          |      25,228,938 |
| hi  | Hindi           |      15,127,900 |
| hr  | Croatian        |      18,797,643 |
| hu  | Hungarian       |      36,435,409 |
| id  | Indonesian      |      70,545,705 |
| ig  | Igbo            |          80,385 |
| ilo | Ilocano         |         335,469 |
| is  | Icelandic       |       8,723,145 |
| it  | Italian         |     146,240,552 |
| ja  | Japanese        |      40,883,733 |
| jv  | Javanese        |         819,280 |
| ko  | Korean          |      19,358,582 |
| la  | Latin           |       1,114,190 |
| lb  | Luxembourgish   |      11,978,495 |
| lt  | Lithuanian      |      23,298,470 |
| lv  | Latvian         |      16,685,969 |
| mg  | Malagasy        |       1,736,359 |
| mk  | Macedonian      |      12,040,173 |
| ml  | Malayalam       |       6,809,956 |
| mr  | Marathi         |       2,874,211 |
| ms  | Malay           |      10,730,648 |
| ne  | Nepali          |         708,316 |
| nl  | Dutch           |     106,695,917 |
| no  | Norwegian       |      47,801,406 |
| oc  | Occitan         |       1,730,828 |
| or  | Odia (Oriya)    |          96,595 |
| pl  | Polish          |      74,070,714 |
| pt  | Portuguese      |     173,743,166 |
| ro  | Romanian        |      55,607,023 |
| ru  | Russian         |     139,937,785 |
| sd  | Sindhi          |       1,717,573 |
| si  | Sinhala         |       6,270,800 |
| sk  | Slovak          |      38,096,241 |
| sl  | Slovenian       |      27,406,782 |
| so  | Somali          |         222,793 |
| sq  | Albanian        |      22,358,158 |
| sr  | Serbian         |      26,510,872 |
| su  | Sundanese       |         271,736 |
| sv  | Swedish         |      77,008,059 |
| sw  | Swahili         |       5,756,664 |
| ta  | Tamil           |       7,291,118 |
| tl  | Tagalog         |       3,113,828 |
| tr  | Turkish         |      47,045,956 |
| uk  | Ukrainian       |      20,240,171 |
| ur  | Urdu            |       6,094,149 |
| vi  | Vietnamese      |      50,092,444 |
| xh  | Xhosa           |      18,980,689 |
| yi  | Yiddish         |         275,076 |
| zh  | Chinese         |      71,383,325 |



```commandline
python3 data.py --source_lang en  --target_lang fr  --min_char_length 32
```
source_lang (str): The source language's id. 

target_lang (str): The target language's id. 

min_char_length (int): The minimum length in terms of characters for train and eval cases. 

