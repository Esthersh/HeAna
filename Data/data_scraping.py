import io
import pandas as pd

import requests
from pypdf import PdfReader

from functools import reduce

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# url
# url = 'https://www.nite.org.il//wp-content/uploads/2023/01/psychometric_winter_2022_acc.pdf'
# other_url = 'https://www.psychometry.co.il/uploaded_files/e2122197062ac17145d7d55634e03ad2.pdf'
# next = 'https://www.psychometry.co.il/uploaded_files/69db6af66892ca0dc4333613ff3f7bc2.pdf'

# Create the binary string html containing the HTML source

# res = requests.get(next, verify=False)
# res.encoding = 'utf-8'
# html = res.text
# res.text.encode("utf8")
# # soup = BeautifulSoup(html)
#
# # soup = bs(res.text,'html.parser')
# # a=soup.findAll('presentation')
# # for tag in a:
# #     print(tag)
#
#
# # print(soup)
#
# f = io.BytesIO(res.content)
# reader = PdfReader(f)
# for page in reader.pages:
# for page, index in enumerate(reader.pages):
#     if 'שאלותאנלוגיות' in page.extract_text() or :

data_dict = {'moed': [], 'base': [], 'option 1': [], 'option 2': [], 'option 3':[], 'option 4':[]}
desc_dict = {'base': [], 'option 1': [], 'option 2': [], 'option 3':[], 'option 4':[], 'correct_ans': []}
str_to_trim = [
               "אין להעתיק או להפיץ בחינה זו או קטעים ממנה בכל צורה ובכל אמצעי, או ללמדה - כולה או חלקים ממנה - בלא אישור בכתב מהמרכז הארצי לבחינות ולהערכהחשיבה מילולית - פרק ראשון",
                "אין להעתיק או להפיץ בחינה זו או קטעים ממנה בכל צורה ובכל אמצעי, או ללמדה - כולה או חלקים ממנה - בלא אישור בכתב מהמרכז הארצי לבחינות ולהערכהחשיבה מילולית - פרק שני",
                "בפרק זה"
               ]
option_trims = ['תשובה (', '(', ')', '1', '2', '3', '4', '.', 'התשובה נפסלת', 'זו התשובה הנכונה']
base_trims = ['1', '2', '3', '4', '5', '6', '.']
correct_ans_trims = ['התשובה', 'הנכונה', 'היא', '(', ')', '.']


def read_url(url, is_solution = False):
    res = requests.get(url, verify=False)
    res.encoding = 'utf-8'
    html = res.text
    res.text.encode("utf8")
    f = io.BytesIO(res.content)
    reader = PdfReader(f)
    if not is_solution:
        extract_analogies(reader)
    else:
        read_soultion(reader)


def to_english(word):
    milon = {'חורף': 'WIN', 'קיץ': 'SUM', 'אביב': 'SPR',
            'סתיו': 'FAL', 'דצמבר': 'DEC', 'ספטמבר': 'SEP',
             'יולי': 'JUL', 'אפריל': 'APR', 'פברואר': 'FEB', 'אוקטובר': 'OCT'}

    return milon[word]


def get_moed(reader):
    seasons = ['חורף', 'קיץ', 'אביב', 'סתיו']
    months = ['דצמבר', 'ספטמבר', 'יולי', 'אפריל', 'פברואר', 'אוקטובר']
    page = reader.pages[0]
    contents = page.extract_text().split('\n')

    for item in contents:
        for season in seasons:
            if season in item:
                return to_english(season) + item[:4]


def extract_analogies(reader):
    moed = get_moed(reader)

    for page_num in [4, 5, 12, 13]:
        page = reader.pages[page_num]
        contents = page.extract_text().split('\n')
        i = 5
        for item in contents:
            for s in str_to_trim:
                item = item.replace(s, '')

            if item == '':
                continue

            if i > 4 and len(item) > 0 and item[-1].isnumeric() and int(item[-1]) <= 6: # we found base
                base = item[0:-1].strip(" -").replace(" : ", ":")
                # print()
                # print(base)
                data_dict['base'].append(base)
                data_dict['moed'].append(moed)
                i = 0

            if i >= 1 and i <= 4: # for all other options
                for trim in option_trims:
                    item = item.replace(trim, '')
                if item == '':
                    data_dict[f'option {i}'].append('FILL THIS OUT')
                else:
                    data_dict[f'option {i}'].append(item)

            i += 1
            # if len(item) > 3 and item[-1] == "(" and item[-3] == ")":
            #     i += 1
            #     if i > 4:
            #         break
            #     # print(item)
            #     option = item[0:-4]
            #     # print(option)
            #     data_dict[f'option {i}'].append(option)


def read_soultion(reader):
    for page_num in [0, 1, 6, 7]:
        page = reader.pages[page_num]
        contents = page.extract_text().split('\n')
        i = 6
        for item in contents:
            print(item)

            if len(item) > 0 and item[0].isnumeric() and int(item[0]) <= 6:  # we found base
                base = item
                for s in base_trims:
                    base = base.replace(s, '')
                base.strip()

                if base == '':
                    desc_dict['base'].append('FILL THIS OUT')
                else:
                    desc_dict['base'].append(base)
                i = 0

            if i >= 1 and i <= 4:  # for all other options
                option = item

                for trim in option_trims:
                    option = option.replace(trim, '')
                option = option.replace(':', '')
                if item == '':
                    desc_dict[f'option {i}'].append('FILL THIS OUT')
                else:
                    if not 'תשובה (' in item:  # could be the rest of the prev option
                        i -= 1
                        prev = desc_dict[f'option {i}'].pop()
                        option = prev + option
                        desc_dict[f'option {i}'].append(option)
                    else: # all good i hope...
                        desc_dict[f'option {i}'].append(option)

            if i == 5: # correct answer value
                for trim in correct_ans_trims:
                    item = item.replace(trim, '')
                item = item.strip()
                try:
                    correct_ans = int(item)
                    desc_dict['correct_ans'].append(str(correct_ans))
                except:
                    desc_dict['correct_ans'].append('FILL THIS OUT')

            i += 1

def remove_redundant_txt(s):
    to_remove = ['\u202a', '\u202b', '\u202c', '\n']

    for item in to_remove:
        s = s.replace(item, '')
    s = s.strip()

    return s


def clean_txt_file(filename, new_filename):
    f1 = open(filename, encoding='utf-8', mode='r')
    new_lines = []
    for line in f1:
        line = remove_redundant_txt(line) + '\n'
        new_lines.append(line)
    f2 = open(new_filename, encoding='utf-8', mode='w')
    f1.writelines(f2)
    f1.close()
    f2.close()


def read_txt_solution():
    with open('solutions/sol4.txt', encoding='utf-8') as contents:
        i = 6
        for item in contents:
            item = remove_redundant_txt(item)
            print(item)

            if len(item) > 0 and \
                item[1].isnumeric() and not item[2].isnumeric() \
                and int(item[1]) <= 6:  # we found base
                base = item
                for s in base_trims:
                    base = base.replace(s, '')
                base.strip()

                if base == '':
                    desc_dict['base'].append('FILL THIS OUT')
                else:
                    desc_dict['base'].append(base)
                i = 0

            if i >= 1 and i <= 4:  # for all other options
                option = item

                for trim in option_trims:
                    option = option.replace(trim, '')
                option = option.replace(':', '')

                if option == '':
                    desc_dict[f'option {i}'].append('FILL THIS OUT')
                else:
                    if not 'תשובה (' in item:  # could be the rest of the prev option
                        i -= 1
                        prev = desc_dict[f'option {i}'].pop()
                        option = prev + option
                        desc_dict[f'option {i}'].append(option)
                    else:  # all good i hope...
                        desc_dict[f'option {i}'].append(option)

                if 'התשובה הנכונה' in item:
                    desc_dict['correct_ans'].append(str(i))



            # if i == 5:  # correct answer value
            #     for trim in correct_ans_trims:
            #         item = item.replace(trim, '')
            #     item = item.strip()
            #     try:
            #         correct_ans = int(item)
            #         desc_dict['correct_ans'].append(str(correct_ans))
            #     except:
            #         desc_dict['correct_ans'].append('FILL THIS OUT')

            i += 1



read_txt_solution()


# with open('sol_urls.txt') as f:
#     for url in f:
#         read_url(url.strip(), True)

with open('DataScraping/urls.txt') as f:
    for url in f:
        read_url(url.strip())

# pad data lists to be the same length
max_len = 0
for key in data_dict.keys():
    if len(data_dict.get(key)) > max_len:
        max_len = len(data_dict.get(key))

for key in data_dict.keys():
    while len(data_dict.get(key)) < max_len:
        data_dict.get(key).append('')

max_len = 0
for key in desc_dict.keys():
    if len(desc_dict.get(key)) > max_len:
        max_len = len(desc_dict.get(key))

for key in desc_dict.keys():
    while len(desc_dict.get(key)) < max_len:
        desc_dict.get(key).append('')

print(pd.DataFrame.from_dict(desc_dict))


df = pd.DataFrame.from_dict(data_dict)
df.to_csv('analogs3.csv', encoding='utf-8-sig')
print(pd.DataFrame.from_dict(data_dict))
