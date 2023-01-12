import os
from whacc import utils
import optuna
import lightgbm as lgb
from sklearn import metrics
from whacc import analysis
import numpy as np
import shutil




class retrain_LGBM():
    def __init__(self, model_base_dir, study_name, tvt_x, tvt_y, tvt_fn, tvt_w, load_optuna_if_exists=False):
        global GLOBALS
        GLOBALS = dict()
        # GLOBALS = GLOBALSin

        # bd = '/Users/phil/Desktop/tmp_mod_test/'
        # study_name = 'my_custom_optuna_models_test_V1'

        GLOBALS['mod_dir'] = model_base_dir + os.sep + study_name + os.sep

        GLOBALS['num_optuna_trials'] = 20  ########  20  3
        GLOBALS['early_stopping_rounds'] = 100  ########  100 10
        GLOBALS['num_iterations'] = 10000 ########  10000 5
        ######################## USER SETTING AFFECTING THE FINAL EVAL RESULTS
        GLOBALS['edge_threshold'] = 5
        GLOBALS['thresholds'] = np.linspace(0.4, .8, 5)
        GLOBALS['smooth_by'] = 5
        ########################
        GLOBALS['tvt_x'] = tvt_x
        GLOBALS['tvt_y'] = tvt_y
        GLOBALS['tvt_fn'] = tvt_fn

        GLOBALS['load_optuna_if_exists'] = load_optuna_if_exists

        GLOBALS['tvt_w'] = tvt_w
        ########################
        GLOBALS['study_name'] = study_name
        GLOBALS['storage_dir'] = GLOBALS['mod_dir']
        GLOBALS['basic_info'] = dict()
        print("""you can change parameters using the name of the class variable plus '.GLOBALS['num_optuna_trials']' for example. see list below for class info""")
        utils.info(GLOBALS)
        self.GLOBALS = GLOBALS

    def set_global(self):
        GLOBALS = self.GLOBALS

    def train_model(self):
        self.set_global()

        mod_dir = GLOBALS['mod_dir']
        if os.path.isdir(mod_dir):
            if GLOBALS['load_optuna_if_exists']:
                print('Attempting to load previous optuna training file...')
            else:
                inp = input('directory exists already, type "delete" to delete existing training folder, cancel or type anything else to stop training')
                if inp.lower() == 'delete':
                    shutil.rmtree(mod_dir)
                    print('directory was deleted and remade ready to run!')
                else:
                    assert False, "directory already exists stopping training "

        utils.make_path(mod_dir)

    # if __name__ == "__main__":
        study_name = GLOBALS['study_name']
        storage_dir = GLOBALS['storage_dir']

        storage = f"sqlite:///{storage_dir}/OPTUNA_SAVE_{study_name}.db"

        study = optuna.create_study(study_name=study_name, storage=storage, direction="maximize",
                                    load_if_exists=GLOBALS['load_optuna_if_exists'])

        study.optimize(self.objective, n_trials=GLOBALS['num_optuna_trials'], timeout=None, n_jobs=1,
                       show_progress_bar=True)

        print("Number of finished trials: {}".format(len(study.trials)))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: {}".format(trial.value))

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))


    @staticmethod
    def objective(trial):
        def reset_metrics():
            def callback(env):
                mod_dir = GLOBALS['mod_dir']
                edge_threshold = GLOBALS['edge_threshold']
                thresholds = GLOBALS['thresholds']
                smooth_by = GLOBALS['smooth_by']

                real = GLOBALS['tvt_y'][2]
                touch_count = np.sum(np.diff(real) == 1)
                frame_nums = GLOBALS['tvt_fn'][2]
                yhat = env.model.predict(GLOBALS['tvt_x'][2])
                yhat_proba = utils.smooth(yhat, smooth_by)
                df = analysis.thresholded_error_types(real, yhat_proba, edge_threshold=edge_threshold,
                                                      frame_num_array=frame_nums, thresholds=thresholds)
                df2 = df.iloc[:, :-2] / touch_count
                x = df2.sum(axis=1)
                err = x.min()

                GLOBALS['all_errors_test'].append(df)
                GLOBALS['metrics_test']['auc'].append(metrics.roc_auc_score(real, yhat))
                GLOBALS['metrics_test']['touch_count_errors_per_touch'].append(err)
                GLOBALS['metrics_test']['best_threshold'].append(thresholds[x.argmin()])

                # do the sam for the trianing data
                real = GLOBALS['tvt_y'][0]
                touch_count = np.sum(np.diff(real) == 1)
                frame_nums = GLOBALS['tvt_fn'][0]
                yhat = env.model.predict(GLOBALS['tvt_x'][0])
                yhat_proba = utils.smooth(yhat, smooth_by)
                df = analysis.thresholded_error_types(real, yhat_proba, edge_threshold=edge_threshold,
                                                      frame_num_array=frame_nums, thresholds=thresholds)
                df2 = df.iloc[:, :-2] / touch_count
                x = df2.sum(axis=1)
                err = x.min()

                GLOBALS['all_errors_train'].append(df)
                GLOBALS['metrics_train']['auc'].append(metrics.roc_auc_score(real, yhat))
                GLOBALS['metrics_train']['touch_count_errors_per_touch'].append(err)
                GLOBALS['metrics_train']['best_threshold'].append(thresholds[x.argmin()])

            return callback
        def thresholded_error_types(yhat, dmat):
            ######################## SHOULDNT CHANGE
            frame_nums_val = GLOBALS['tvt_fn'][1]
            mod_dir = GLOBALS['mod_dir']
            higher_is_better = False
            edge_threshold = GLOBALS['edge_threshold']
            thresholds = GLOBALS['thresholds']
            smooth_by = GLOBALS['smooth_by']
            ######################## SHOULDNT CHANGE -- CALC TCerr
            yhat_proba = utils.smooth(yhat, smooth_by)
            real = dmat.get_label() if isinstance(dmat, lgb.Dataset) else dmat
            touch_count = np.sum(np.diff(real) == 1)
            #     with HiddenPrints():
            df = analysis.thresholded_error_types(real, yhat_proba,
                                                  edge_threshold=edge_threshold,
                                                  frame_num_array=frame_nums_val,
                                                  thresholds=thresholds)
            df2 = df.iloc[:, :-2] / touch_count
            x = df2.sum(axis=1)
            err = x.min()
            ######################## THE REST IS SAVING AND RETURN FOR LGBM EVAL
            GLOBALS['all_errors'].append(df)
            GLOBALS['metrics_val']['auc'].append(metrics.roc_auc_score(real, yhat))
            GLOBALS['metrics_val']['touch_count_errors_per_touch'].append(err)
            GLOBALS['metrics_val']['best_threshold'].append(thresholds[x.argmin()])
            if 'basic_info' not in list(GLOBALS.keys()):
                GLOBALS['basic_info'] = dict()
            GLOBALS['basic_info']['threshold'] = thresholds
            GLOBALS['basic_info']['higher_is_better'] = higher_is_better
            GLOBALS['basic_info']['edge_threshold'] = edge_threshold
            GLOBALS['basic_info']['touch_count'] = touch_count
            GLOBALS['basic_info']['real_human_predictions'] = real
            GLOBALS['basic_info']['smooth_by'] = smooth_by
            GLOBALS['basic_info']['num_optuna_trials'] = GLOBALS['num_optuna_trials']
            GLOBALS['basic_info']['early_stopping_rounds'] = GLOBALS['early_stopping_rounds']
            GLOBALS['basic_info']['num_iterations'] = GLOBALS['num_iterations']
            GLOBALS['basic_info']['num_optuna_trials'] = GLOBALS['num_optuna_trials']
            return "touch_count_errors", err, higher_is_better

        d = utils.load_feature_data()
        names = np.asarray(d['full_feature_names_and_neuron_nums'])
        final_feature_names_USE = list(names[d['final_selected_features_bool']])
        if len(final_feature_names_USE) != GLOBALS['tvt_x'][0].shape[-1]:
            print('not using the 2105 features dataset, removing feature names and replacing with numbers')
            final_feature_names_USE = [str(k).zfill(4) for k in np.arange(GLOBALS['tvt_x'][0].shape[-1])]

        train_DATA = lgb.Dataset(GLOBALS['tvt_x'][0], label=GLOBALS['tvt_y'][0],
                                 feature_name=final_feature_names_USE, weight=GLOBALS['tvt_w'][0])
        val_DATA = lgb.Dataset(GLOBALS['tvt_x'][1], label=GLOBALS['tvt_y'][1],
                               feature_name=final_feature_names_USE, reference=train_DATA,
                               weight=GLOBALS['tvt_w'][1])

        param = {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "early_stopping_rounds": GLOBALS['early_stopping_rounds'],
            # "num_boost_round": GLOBALS['num_iterations'],
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 2, 256),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            #         'first_metric_only' : True,
        }

        ######################## INIT THE LISTS FOR SAVING LATER

        GLOBALS['metrics_val'] = dict()
        GLOBALS['metrics_val']['auc'] = []
        GLOBALS['metrics_val']['touch_count_errors_per_touch'] = []
        GLOBALS['metrics_val']['best_threshold'] = []
        GLOBALS['all_errors'] = []

        GLOBALS['metrics_test'] = dict()
        GLOBALS['metrics_test']['auc'] = []
        GLOBALS['metrics_test']['touch_count_errors_per_touch'] = []
        GLOBALS['metrics_test']['best_threshold'] = []
        GLOBALS['all_errors_test'] = []

        GLOBALS['metrics_train'] = dict()
        GLOBALS['metrics_train']['auc'] = []
        GLOBALS['metrics_train']['touch_count_errors_per_touch'] = []
        GLOBALS['metrics_train']['best_threshold'] = []
        GLOBALS['all_errors_train'] = []

        gbm = lgb.train(
            param,
            train_DATA,
            num_boost_round=GLOBALS['num_iterations'],
            valid_sets=[val_DATA],
            feval=thresholded_error_types,
            callbacks=[reset_metrics()],
        )

        preds = gbm.predict(GLOBALS['tvt_x'][1])
        fpr, tpr, thresholds_auc = metrics.roc_curve(GLOBALS['tvt_y'][1], preds, pos_label=1)
        AUC_out = metrics.auc(fpr, tpr)

        # save the model
        mod_num = str(len(utils.get_files(GLOBALS['mod_dir'], '*_model.pkl'))).zfill(3)
        fn = GLOBALS['mod_dir'] + mod_num + '_model.pkl'
        utils.save_obj(gbm, fn)

        d = dict()
        d['metrics'] = GLOBALS['metrics_val']
        d['all_errors'] = GLOBALS['all_errors']
        d['metrics_test'] = GLOBALS['metrics_test']
        d['all_errors_test'] = GLOBALS['all_errors_test']

        fn = GLOBALS['mod_dir'] + mod_num + '_model_results.pkl'
        utils.save_obj(d, fn)

        if mod_num == '000':
            fn = GLOBALS['mod_dir'] + 'basic_info.pkl'
            utils.save_obj(GLOBALS['basic_info'], fn)

        return AUC_out

