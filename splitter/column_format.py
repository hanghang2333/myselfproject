import cv2
import re
# from splitter import split_from_url_with_pixel


def column_format_by_pixel(pic_list, pixel_list):
    """
    format each column in each line.
    input: a 2D matrix, each element in matrix is a picture.
    output: a 2D matrix, each line has the same number of elements.
            if there is no element in that place, we use None replaced.
    """
    # delete the lines that have less than 2 elements
    wrong_lines = []
    for i, line in enumerate(pic_list):
        if len(line) < 3:
            wrong_lines.append(i)
    for i in wrong_lines[::-1]:
        del pic_list[i]
        del pixel_list[i]

    # main funtion

    # judge whether the length is less than 2
    if len(pixel_list) < 2:
        return pic_list

    # format the column line by line
    column_list = pixel_list[0]
    for line in pixel_list[1:]:
        for i, pixel in enumerate(line):
            for j, column in enumerate(column_list):
                if abs(pixel - column) < 15:
                    column_list[j] = pixel
                    break
            else:
                column_list.append(pixel)
    print(len(column_list))
    print(column_list)


def column_format_by_rule(str_list):
    """
    format the column by the string of picture recognized by Lihang.
    receive a string list with 2D, each element represents the content of a piece.
    return a 2D list with the same length in each dimension.
    """
    examine_pattern = re.compile('[\u4e00-\u9fa5]')
    value_pattern = re.compile('[0-9]')
    range_pattern = re.compile('[\-]')

    examine_list = []
    for line in str_list:
        examine_line = [None, None, None]
        for element in line:
            if len(re.findall(range_pattern, element)) > 0:
                if examine_line[2] is None:
                    examine_line[2] = element
            elif len(re.findall(value_pattern, element)) > 0:
                if examine_line[1] is None:
                    examine_line[1] = element
            elif len(re.findall(examine_pattern, element)) > 1:
                if examine_line[0] is None:
                    examine_line[0] = element
        examine_list.append(examine_line)
    return examine_list


def test():
    # file_path = 'http://imgs.hh-medic.com/order/D2017062714390659744/2017-06-27/ntHtH2.jpeg'
    # pic_list, pixel_list = split_from_url_with_pixel(file_path)
    # column_format(pic_list, pixel_list)

    # We use the csv file produced by Lihang to evaluate the preformance of
    # the method of column format by rule.
    def read_str_list(src):
        """
        read csv file and save it as a 2D list.
        Because there is not a format csv file, we cannot read it by pandas.
        So we read it line by line.
        """
        str_list = []
        with open(src, 'r', encoding='utf8') as data:
            for line in data:
                elements = line.rstrip().split(' ')
                str_list.append(elements)
        return str_list

    result_src = '../splitter/data/result/6.csv'

    # save the pieces of pictures that splitted to check the performance.
    def save_pic():
        from .splitter import split_from_url
        import os
        pic_list = split_from_url('../splitter/data/pic_data/6.jpg')
        test_root_path = '../../splitter/data/test'
        if not os.path.exists(test_root_path):
            os.mkdir(test_root_path)
        for i, pic_line in enumerate(pic_list):
            line_path = os.path.join(test_root_path, str(i))
            if not os.path.exists(line_path):
                os.mkdir(line_path)
            for j, pic in enumerate(pic_line):
                pic_path = os.path.join(line_path, str(j) + '.png')
                print(pic_path)
                cv2.imwrite(pic_path, pic)

    str_list = read_str_list(result_src)
    examine_list = column_format_by_rule(str_list)
    for str_line, examine_line in zip(str_list, examine_list):
        print(examine_line)
