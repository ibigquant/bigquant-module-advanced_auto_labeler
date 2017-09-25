import logbook
import numpy as np
import pandas as pd

from bigdata.api.datareader import D
import bigexpr
import biglearning.api.tools as T
from biglearning.module2.common.data import DataSource, Outputs
import biglearning.module2.common.interface as I
from bigshared.common.utils import extend_class_methods
from biglearning.module2.common.utils import smart_list

bigquant_cacheable = True
log = logbook.Logger('advanced_auto_labeler')


DEFAULT_LABEL_EXPR = """# #号开始的表示注释
# 0. 每行一个，顺序执行，从第二个开始，可以使用label字段
# 1. 可用数据字段见 https://bigquant.com/docs/data_history_data.html
#   添加benchmark_前缀，可使用对应的benchmark数据
# 2. 可用操作符和函数见 `表达式引擎 <https://bigquant.com/docs/big_expr.html>`_

# 计算收益：5日收盘价(作为卖出价格)除以明日开盘价(作为买入价格)
shift(close, -5) / shift(open, -1)

# 极值处理：用1%和99%分位的值做clip
clip(label, all_quantile(label, 0.01), all_quantile(label, 0.99))

# 将分数映射到分类，这里使用20个分类
all_wbins(label, 20)

# 过滤掉一字涨停的情况 (设置label为NaN，在后续处理和训练中会忽略NaN的label)
where(shift(high, -1) == shift(low, -1), NaN, label)
"""


# 模块接口定义
bigquant_category = '数据标注'
bigquant_friendly_name = '自动数据标注'
bigquant_doc_url = 'https://bigquant.com/docs/module_advanced_auto_labeler.html'


class BigQuantModule:
    def __init__(
        self,
        instruments: I.port('证券代码列表', specific_type_name='列表|DataSource'),
        start_date: I.str('开始日期，示例 2017-02-12'),
        end_date: I.str('结束日期，示例 2017-02-12'),
        label_expr: I.code('标注表达式，可以使用多个表达式，顺序执行，从第二个开始，可以使用label字段。可用数据字段见 https://bigquant.com/docs/data_history_data.html，添加benchmark_前缀，可使用对应的benchmark数据。可用操作符和函数见 `表达式引擎 <https://bigquant.com/docs/big_expr.html>`_', default=DEFAULT_LABEL_EXPR, specific_type_name='列表'),
        benchmark: I.str('基准指数，如果给定，可以使用 benchmark_* 变量') = '000300.SHA',
        drop_na_label: I.bool('删除无标注数据') = True,
        cast_label_int: I.bool('将标注转换为整数') = True,
        user_functions: I.code('自定义表达式函数，用户自定义表达式函数，参考 文档-表达式引擎 https://bigquant.com/docs/big_expr.html', I.code_python, specific_type_name='函数') = '') -> [
            I.port('标注数据', 'data')
        ]:
        '''
        自动数据标注：advanced_auto_labeler可以使用表达式，对数据做任何标注。比如基于未来给定天数的收益/波动率等数据，来实现对数据做自动标注。
        '''

        self.__instruments = smart_list(instruments)
        self.__start_date = start_date
        self.__end_date = end_date
        self.__label_expr = smart_list(label_expr)
        self.__benchmark = benchmark
        self.__drop_na_label = drop_na_label
        self.__cast_label_int = cast_label_int
        self.__user_functions = user_functions

    def __load_data(self):
        BENCHMARK_FEATURE_PREFIX = 'benchmark_'

        general_features = []
        for expr in self.__label_expr:
            general_features += bigexpr.extract_variables(expr)
            if len(general_features) == 1:
                if 'label' in general_features[0]:
                    raise Exception('label变量不能使用在第一个表达式上')
        if len(general_features) < 1:
            raise Exception('没有标注表达式')
        general_features = set(general_features)
        if 'label' in general_features:
            general_features.remove('label')
        general_features = list(general_features)

        instrument_general_features = [f for f in general_features if not f.startswith(BENCHMARK_FEATURE_PREFIX)]
        if not instrument_general_features:
            raise Exception('在表达式中没有发现需要加载的字段: %s' % self.__label_expr)
        history_data = D.history_data(
            instruments=self.__instruments,
            start_date=self.__start_date,
            end_date=self.__end_date,
            fields=['date', 'instrument', 'amount'] + instrument_general_features)

        benchmark_general_features = [f[len(BENCHMARK_FEATURE_PREFIX):] for f in general_features if f.startswith(BENCHMARK_FEATURE_PREFIX)]
        if benchmark_general_features:
            benchmark_data = D.history_data(
                instruments=self.__benchmark,
                start_date=self.__start_date,
                end_date=self.__end_date,
                fields=['date', 'instrument'] + benchmark_general_features)
            benchmark_data.columns = [c if c in ['date'] else BENCHMARK_FEATURE_PREFIX + '%s' % c \
                                      for c in benchmark_data.columns]
            history_data = history_data.merge(benchmark_data, on='date', how='left')

        history_data = history_data[history_data.amount > 0]
        log.info('加载历史数据: %s 行' % len(history_data))
        return history_data

    def run(self):
        df = self.__load_data()
        log.info('开始标注 ..')

        for expr in self.__label_expr:
            df['label'] = bigexpr.evaluate(df, expr, self.__user_functions)

        if self.__drop_na_label:
            df = df.dropna(subset=['label'])

        if self.__cast_label_int:
            df['label'] = df['label'].astype(int)
            if df['label'].min() < 0:
                raise Exception('label必须为非负整数，错误数据 label=%s' % df['label'].min())

        df.reset_index(drop=True, inplace=True)
        df.columns = [col if col in ['date', 'instrument', 'label'] else 'm:' + col for col in df.columns]

        data = DataSource.write_df(df)
        outputs = Outputs(data=data, cast_label_int=self.__cast_label_int)

        return outputs


def bigquant_postrun(outputs) -> [
        I.doc('绘制标注数据分布', 'plot_label_counts'),
    ]:
    def plot_label_counts(self):
        df = self.data.read_df()
        if self.cast_label_int:
            label_counts = sorted(df['label'].value_counts().to_dict().items())
            df = pd.DataFrame(label_counts, columns=['label', 'count']).set_index('label')
            T.plot(df, title='数据标注分布', double_precision=0, chart_type='column')
        else:
            bin_counts = np.histogram(df['label'], bins=20)
            label_counts = pd.DataFrame(data=list(bin_counts)).transpose()
            label_counts.columns = ['count', 'label']
            T.plot(label_counts, x='label', y=['count'], chart_type='column', title='label', options={'series': [{
                'pointPadding': 0,
                'groupPadding': 0,
                'pointPlacement': 'between'
            }]})
    extend_class_methods(outputs, plot_label_counts=plot_label_counts)

    return outputs


if __name__ == '__main__':
    # 测试代码
    pass