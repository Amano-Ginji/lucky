#!/usr/bin/python

import sys
import re
import random

from datetime import datetime

from collections import deque
from collections import defaultdict

import logging

import argparse

HASH_SIZE = 1000000

if len(sys.argv) < 5:
    logging.error("Usage: cat input | python {} output1 output2 ratio mode".format(sys.argv[0]))
    exit(1)

output_file_1 = sys.argv[1]
output_file_2 = sys.argv[2]
ratio = float(sys.argv[3])
mode = int(sys.argv[4])

fout1 = open(output_file_1, 'w')
fout2 = open(output_file_2, 'w')

# key: code
# value: cnt
dict_dist_0days = defaultdict()
dict_dist_1days = defaultdict()
dict_dist_7days = defaultdict()
dict_dist_30days = defaultdict()
dict_dist_365days = defaultdict()

# (code, date)
queue_codes_0days = deque()
queue_codes_1days = deque()
queue_codes_7days = deque()
queue_codes_30days = deque()
queue_codes_365days = deque()

def date_diff(d1, d2):
    ''' format: %Y%m%d '''
    d1 = datetime.strptime(d1, "%Y%m%d")
    d2 = datetime.strptime(d2, "%Y%m%d")
    return (d2-d1).days

def update_queue_dict(dict_dist, queue_codes, code, dt, days):
    while len(queue_codes)>0 and date_diff(queue_codes[0][1], dt) > days:
        c1, d1 = queue_codes.popleft()
        if c1 in dict_dist:
            dict_dist[c1] -= 1
    queue_codes.append((code, dt))
    if code not in dict_dist:
        dict_dist[code] = 0
    dict_dist[code] += 1

def hash_feat(slot_name, value, hash_size=HASH_SIZE):
    return (hash("{}-{}".format(slot_name, value))%hash_size)+1

def feature_extract_hash(dt, year, month, day, sid):
    features = set()
    features.add(hash_feat("month", month))
    features.add(hash_feat("day", day))
    features.add(hash_feat("sid", sid))
    features.add(hash_feat("month_day", month+day))
    features.add(hash_feat("month_sid", month+sid))
    features.add(hash_feat("day_sid", day+sid))
    features.add(hash_feat("month_day_sid", month+day+sid))

    if len(queue_codes_0days) > 0:
        for code,cnt in dict_dist_0days.items():
            slot_name = "codes_1days_{}".format(code)
            features.add(hash_feat(slot_name, cnt))
    if len(queue_codes_1days) > 0:
        for code,cnt in dict_dist_1days.items():
            slot_name = "codes_1days_{}".format(code)
            features.add(hash_feat(slot_name, cnt))
    if len(queue_codes_7days) > 0:
        features.add(hash_feat("last_code", queue_codes_7days[-1][0]))
        for code,cnt in dict_dist_7days.items():
            slot_name = "codes_7days_{}".format(code)
            features.add(hash_feat(slot_name, cnt))
            #features.add(hash_feat(slot_name, "{:.6f}".format(float(cnt)/len(queue_codes_7days))))
    if len(queue_codes_30days) > 0:
        for code,cnt in dict_dist_30days.items():
            slot_name = "codes_30days_{}".format(code)
            features.add(hash_feat(slot_name, cnt))
            #features.add(hash_feat(slot_name, "{:.6f}".format(float(cnt)/len(queue_codes_30days))))
    if len(queue_codes_365days) > 0:
        for code,cnt in dict_dist_365days.items():
            slot_name = "codes_365days_{}".format(code)
            features.add(hash_feat(slot_name, cnt))
            #features.add(hash_feat(slot_name, "{:.6f}".format(float(cnt)/len(queue_codes_365days))))

    return " ".join(["{}:1".format(feat) for feat in sorted(features)])

dict_feature_name_index = {
    'month': 0,
    'day': 1,
    'sid': 2,
    'month_day': 3,
    'month_sid': 4,
    'day_sid': 5,
    'month_day_sid': 6,
    'last_code': 7,
    'codes_0days': range(8, 18),
    'codes_1days': range(18, 28),
    'codes_7days': range(28, 38),
    'codes_30days': range(38, 48),
    'codes_365days': range(48, 58)
}
feature_num = 58

def update_feature(features, idx, value):
    features[idx] = value

def feature_extract(dt, year, month, day, sid):
    features = [-1]*feature_num

    update_feature(features, dict_feature_name_index['month'], month)
    update_feature(features, dict_feature_name_index['day'], day)
    update_feature(features, dict_feature_name_index['sid'], sid)
    update_feature(features, dict_feature_name_index['month_day'], month+day)
    update_feature(features, dict_feature_name_index['month_sid'], month+sid)
    update_feature(features, dict_feature_name_index['day_sid'], day+sid)
    update_feature(features, dict_feature_name_index['month_day_sid'], month+day+sid)

    if len(queue_codes_0days) > 0:
        update_feature(features, dict_feature_name_index['last_code'], queue_codes_7days[-1][0])
        for code in range(0, 10):
            cs = str(code)
            cnt = dict_dist_0days[cs] if cs in dict_dist_0days else 0
            update_feature(features, dict_feature_name_index['codes_0days'][code], cnt)
    if len(queue_codes_1days) > 0:
        for code in range(0, 10):
            cs = str(code)
            cnt = dict_dist_1days[cs] if cs in dict_dist_1days else 0
            update_feature(features, dict_feature_name_index['codes_1days'][code], cnt)
    if len(queue_codes_7days) > 0:
        for code in range(0, 10):
            cs = str(code)
            cnt = dict_dist_7days[cs] if cs in dict_dist_7days else 0
            update_feature(features, dict_feature_name_index['codes_7days'][code], cnt)
    if len(queue_codes_30days) > 0:
        for code in range(0, 10):
            cs = str(code)
            cnt = dict_dist_30days[cs] if cs in dict_dist_30days else 0
            update_feature(features, dict_feature_name_index['codes_30days'][code], cnt)
    if len(queue_codes_365days) > 0:
        for code in range(0, 10):
            cs = str(code)
            cnt = dict_dist_365days[cs] if cs in dict_dist_365days else 0
            update_feature(features, dict_feature_name_index['codes_365days'][code], cnt)

    return ",".join([str(feat) for feat in features])

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue

    date_sid, c1, c2, c3, c4, c5 = re.split("[\t,]", line)

    # base info
    dt = date_sid[0:8]
    year = date_sid[0:4]
    month = date_sid[4:6]
    day = date_sid[6:8]
    sid = "{:0>3}".format(date_sid[8:].rstrip())

    # feature extraction
    if mode == 1:
        feat_pairs = feature_extract(dt, year, month, day, sid)
    else:
        feat_pairs = feature_extract_hash(dt, year, month, day, sid)

    # format instance
    sep = " " if mode==0 else ","
    sample = c5+sep+feat_pairs
    if random.random() < ratio:
        fout1.write(sample)
        fout1.write("\n")
    else:
        fout2.write(sample)
        fout2.write("\n")

    # update queue and dict
    update_queue_dict(dict_dist_0days, queue_codes_0days, c5, dt, 0)
    update_queue_dict(dict_dist_1days, queue_codes_1days, c5, dt, 1)
    update_queue_dict(dict_dist_7days, queue_codes_7days, c5, dt, 7)
    update_queue_dict(dict_dist_30days, queue_codes_30days, c5, dt, 30)
    update_queue_dict(dict_dist_365days, queue_codes_365days, c5, dt, 365)

fout1.close()
fout2.close()


