"""
[Mission 1]
Based on the traditional(or conventional) iVT filter process, customize it!

Step 1)
Raw Gaze Point --> Raw Fixation

Step 2)
Raw Fixation --> Corrected Fixation
"""


def fix(red_dot, wordAoi):
    """
    Description:

    :param red_dot:
    :param wordAoi:
    :return:
    """
    # TODO: Customization needed!
    result = red_dot
    return result


def rm_outlier(fixation):
    """
    Description:

    :param fixation:
    :return:
    """
    # TODO: Customization needed!
    result = fixation
    return result


def run(red_dot, wordAoi):
    """
    Description:

    :param red_dot:
    :param wordAoi:
    :return:
    """
    fixation = fix(red_dot, wordAoi)
    result = rm_outlier(fixation)
    return result


