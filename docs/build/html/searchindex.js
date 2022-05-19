Search.setIndex({docnames:["index","inference","train","utils"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":5,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,sphinx:56},filenames:["index.rst","inference.rst","train.rst","utils.rst"],objects:{"ap.inference":[[1,0,0,"-","server"]],"ap.inference.server":[[1,1,1,"","TopicModelInferenceServiceImpl"]],"ap.inference.server.TopicModelInferenceServiceImpl":[[1,2,1,"","GetDocumentsEmbedding"]],"ap.train":[[2,0,0,"-","data_manager"],[2,0,0,"-","server"],[2,0,0,"-","trainer"]],"ap.train.data_manager":[[2,1,1,"","ModelDataManager"],[2,3,1,"","NoTranslationException"]],"ap.train.data_manager.ModelDataManager":[[2,2,1,"","generate_batches_balanced_by_rubric"],[2,2,1,"","get_modality_distribution"],[2,2,1,"","load_train_data"],[2,2,1,"","update_config"],[2,2,1,"","update_ds_metrics"],[2,2,1,"","write_new_docs"]],"ap.train.server":[[2,1,1,"","TopicModelTrainServiceImpl"]],"ap.train.server.TopicModelTrainServiceImpl":[[2,2,1,"","AddDocumentsToModel"],[2,2,1,"","StartTrainTopicModel"],[2,2,1,"","TrainTopicModelStatus"],[2,2,1,"","UpdateModelConfiguration"]],"ap.train.trainer":[[2,1,1,"","ModelTrainer"]],"ap.train.trainer.ModelTrainer":[[2,2,1,"","generate_model_name"],[2,4,1,"","model_scores"],[2,4,1,"","model_scores_value"],[2,2,1,"","set_metrics"],[2,2,1,"","train_model"]],"ap.utils":[[3,0,0,"-","bpe"],[3,0,0,"-","dictionary"],[3,0,0,"-","emb_metrics"],[3,0,0,"-","general"],[3,0,0,"-","rank_metric"],[3,0,0,"-","search_quality"],[3,0,0,"-","subsamples_creator"],[3,0,0,"-","vowpal_wabbit"],[3,0,0,"-","vowpal_wabbit_bpe"]],"ap.utils.bpe":[[3,5,1,"","load_bpe_models"]],"ap.utils.dictionary":[[3,5,1,"","get_num_entries"],[3,5,1,"","limit_classwise"]],"ap.utils.emb_metrics":[[3,5,1,"","generate_theta"],[3,5,1,"","get_analogy_distribution"],[3,5,1,"","get_cos_distribution"],[3,5,1,"","get_mean_classes_intersection"],[3,5,1,"","get_topic_profile"]],"ap.utils.general":[[3,5,1,"","batch_names"],[3,5,1,"","docs_from_pack"],[3,5,1,"","ensure_directory"],[3,5,1,"","get_modalities"],[3,5,1,"","id_to_str"],[3,5,1,"","recursively_unlink"]],"ap.utils.rank_metric":[[3,5,1,"","quality_of_models"]],"ap.utils.search_quality":[[3,5,1,"","calculate_search_quality"],[3,5,1,"","dump_train_centroids"]],"ap.utils.subsamples_creator":[[3,1,1,"","MakeSubsamples"]],"ap.utils.subsamples_creator.MakeSubsamples":[[3,2,1,"","get_subsamples"]],"ap.utils.vowpal_wabbit":[[3,1,1,"","VowpalWabbit"]],"ap.utils.vowpal_wabbit.VowpalWabbit":[[3,2,1,"","convert_doc"],[3,2,1,"","save_docs"]]},objnames:{"0":["py","module","Python \u043c\u043e\u0434\u0443\u043b\u044c"],"1":["py","class","Python \u043a\u043b\u0430\u0441\u0441"],"2":["py","method","Python \u043c\u0435\u0442\u043e\u0434"],"3":["py","exception","Python \u0438\u0441\u043a\u043b\u044e\u0447\u0435\u043d\u0438\u0435"],"4":["py","property","Python property"],"5":["py","function","Python \u0444\u0443\u043d\u043a\u0446\u0438\u044f"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:exception","4":"py:property","5":"py:function"},terms:{"0x7f038ccde610":[],"0x7fe40f0a58b0":2,"1000":3,"6":3,"\u0430\u043b\u0444\u0430\u0432\u0438\u0442\u043d":0,"\u0430\u0442\u0440\u0438\u0431\u0443\u0442":2,"\u0431\u0430\u0437\u043e\u0432":[1,2],"\u0431\u0430\u0442\u0447":[2,3],"\u0431\u043b\u0438\u0437\u043e\u0441\u0442":3,"\u0431\u043e\u043b":3,"\u0431\u0443\u0434\u0435\u0442":3,"\u0431\u0443\u0434\u0443\u0442":[2,3],"\u0432":[2,3],"\u0432\u0430\u043a":3,"\u0432\u0435\u043a\u0442\u043e\u0440":3,"\u0432\u0435\u0441":[],"\u0432\u0438\u0434":3,"\u0432\u0438\u043a\u0438\u043f\u0435\u0434":2,"\u0432\u043e\u0437\u0432\u0440\u0430\u0437\u0430":[],"\u0432\u043e\u0437\u0432\u0440\u0430\u0449\u0430":[1,2,3],"\u0432\u0441\u0435":[2,3],"\u0432\u0441\u0435\u0433":2,"\u0432\u0441\u0435\u0445":2,"\u0432\u044b\u0433\u0440\u0443\u0437\u043a":3,"\u0432\u044b\u0447\u0438\u0441\u043b\u0435\u043d":3,"\u0432\u044b\u0447\u0438\u0441\u043b\u044f":3,"\u0433\u0434\u0435":[2,3],"\u0433\u0435\u043d\u0435\u0440\u0430\u0442\u043e\u0440":3,"\u0433\u0435\u043d\u0435\u0440\u0438\u0440":[2,3],"\u0433\u0440\u043d\u0442\u0438":[2,3],"\u0434\u0430\u043d":[2,3],"\u0434\u0430\u0442\u0430\u0441\u0435\u0442":2,"\u0434\u0432\u0430":2,"\u0434\u0438\u0440\u0435\u043a\u0442\u043e\u0440":3,"\u0434\u043b\u044f":[2,3],"\u0434\u043e":[2,3],"\u0434\u043e\u0431\u0430\u0432\u043b\u044f":2,"\u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442":[1,2,3],"\u0434\u0440\u0443\u0433":3,"\u0435":3,"\u0435\u0433":3,"\u0435\u043c\u0431\u0435\u0434\u0434\u0438\u043d\u0433":1,"\u0435\u0441\u043b":[2,3],"\u0437\u0430\u0433\u0440\u0443\u0436\u0430":[2,3],"\u0437\u0430\u0433\u0440\u0443\u0437":3,"\u0437\u0430\u0434\u0430":2,"\u0437\u0430\u043f\u0440\u043e\u0441":2,"\u0437\u0430\u043f\u0443\u0441\u043a":2,"\u0437\u0430\u043f\u0443\u0441\u043a\u0430":2,"\u0437\u043d\u0430\u0447\u0435\u043d":[2,3],"\u0438":3,"\u0438\u0437":[2,3],"\u0438\u0437\u043c\u0435\u0440\u0435\u043d":3,"\u0438\u043b":3,"\u0438\u043c":[2,3],"\u0438\u043c\u0435\u043d":[],"\u0438\u043d\u0430\u0447":2,"\u0438\u043d\u0434\u0435\u043a\u0441":3,"\u0438\u043d\u0444\u043e\u0440\u043c\u0430\u0446":2,"\u0438\u0441\u043f\u043e\u043b\u044c\u0437":[1,2],"\u0438\u0441\u043f\u043e\u043b\u044c\u0437\u043e\u0432\u0430":[2,3],"\u0438\u0441\u043f\u043e\u043b\u044c\u0437\u0443\u0435\u043c":3,"\u0438\u0441\u0445\u043e\u0434\u043d":3,"\u0438\u0445":3,"\u043a":3,"\u043a\u0430\u0436\u0434":[2,3],"\u043a\u0430\u0436\u0434\u043e":3,"\u043a\u0430\u0447\u0435\u0441\u0442\u0432":3,"\u043a\u043b\u0430\u0441\u0441":[1,2,3],"\u043a\u043b\u044e\u0447":2,"\u043a\u043e\u043b\u0438\u0447\u0435\u0441\u0442\u0432":[2,3],"\u043a\u043e\u043b\u043b\u0435\u043a\u0446":3,"\u043a\u043e\u043d\u0432\u0435\u0440\u0442\u0438\u0440":3,"\u043a\u043e\u043d\u0442\u0435\u043a\u0441\u0442":[1,2],"\u043a\u043e\u043d\u0444\u0438\u0433":[2,3],"\u043a\u043e\u043d\u0444\u0438\u0433\u0443\u0440\u0430\u0446":2,"\u043a\u043e\u0442\u043e\u0440":[2,3],"\u043a\u0443\u0434":3,"\u043b\u0435\u0436\u0430\u0442":3,"\u043c\u0430\u043a\u0441\u0438\u043c\u0430\u043b\u044c\u043d":3,"\u043c\u0430\u0442\u0440\u0438\u0446":3,"\u043c\u0435\u0440":3,"\u043c\u0435\u0442\u0440\u0438\u043a":[2,3],"\u043c\u043e\u0434\u0430\u043b\u044c\u043d":[2,3],"\u043c\u043e\u0434\u0435\u043b":[2,3],"\u043c\u043e\u0434\u0443\u043b":[0,2],"\u043d\u0430":[2,3],"\u043d\u0430\u0437\u0432\u0430\u043d":3,"\u043d\u0430\u0437\u044b\u0432\u0430\u043d":3,"\u043d\u0430\u043f\u0440\u0438\u043c\u0435\u0440":3,"\u043d\u0435":[1,2,3],"\u043d\u0435\u043e\u0431\u0445\u043e\u0434\u0438\u043c":3,"\u043d\u0435\u0442":3,"\u043d\u043e\u0432":2,"\u043d\u043e\u0440\u043c":3,"\u043e":2,"\u043e\u0431\u043d\u043e\u0432\u043b\u0435\u043d":2,"\u043e\u0431\u043d\u043e\u0432\u043b\u044f":2,"\u043e\u0431\u0440\u0430\u0437":3,"\u043e\u0431\u0443\u0447\u0430":2,"\u043e\u0431\u0443\u0447\u0435\u043d":[2,3],"\u043e\u0431\u044a\u0435\u043a\u0442":2,"\u043e\u0433\u0440\u0430\u043d\u0438\u0447\u0438\u0432\u0430":3,"\u043e\u0434\u0438\u043d\u0430\u043a\u043e\u0432":2,"\u043e\u0434\u043d":3,"\u043e\u0437\u043d\u0430\u0447\u0430":3,"\u043e\u0441\u043d\u043e\u0432\u043d":2,"\u043e\u0442\u0432\u0435\u0442":[1,2],"\u043e\u0442\u043d\u043e\u0441\u0438\u0442\u0435\u043b\u044c\u043d":2,"\u043e\u0446\u0435\u043d\u043a":2,"\u043e\u0447\u0435\u043d":[],"\u043f\u0430\u043f\u043a":3,"\u043f\u0430\u0440\u0430\u043c\u0435\u0442\u0440":[1,2,3],"\u043f\u0435\u0440\u0435\u0434\u0430":2,"\u043f\u0435\u0440\u0435\u0441\u0447\u0438\u0442\u0430":3,"\u043f\u043e":[2,3],"\u043f\u043e\u0434\u0430\u0432\u0431\u043e\u0440\u043e\u043a":3,"\u043f\u043e\u0434\u0432\u044b\u0431\u043e\u0440\u0433\u043a":3,"\u043f\u043e\u0434\u0432\u044b\u0431\u043e\u0440\u043a":3,"\u043f\u043e\u0434\u0432\u044b\u0431\u043e\u0440\u043e\u043a":3,"\u043f\u043e\u0434\u0434\u0435\u0440\u0436\u0430\u043d":2,"\u043f\u043e\u0434\u0440\u043e\u0431\u043d":[],"\u043f\u043e\u0434\u0441\u0447\u0435\u0442":3,"\u043f\u043e\u0434\u0441\u0447\u0438\u0442\u0430":3,"\u043f\u043e\u0438\u0441\u043a":[0,3],"\u043f\u043e\u0438\u0441\u043a\u043e\u0432":3,"\u043f\u043e\u0441\u043b\u0435\u0434\u043d":3,"\u043f\u043e\u0441\u0442\u0440\u043e\u0435\u043d":[2,3],"\u043f\u0440\u0435\u043e\u0431\u0440\u0430\u0437\u043e\u0432\u0430\u043d":3,"\u043f\u0440\u0438\u0437\u043d\u0430\u043a":3,"\u043f\u0440\u0438\u043d\u0438\u043c\u0430":2,"\u043f\u0440\u0438\u0441\u0443\u0442\u0441\u0442\u0432":2,"\u043f\u0440\u043e\u0431\u0435\u043b":3,"\u043f\u0440\u043e\u0438\u0437\u0432\u043e\u0434":3,"\u043f\u0440\u043e\u043c\u0435\u0436\u0443\u0442\u043e\u0447\u043d":3,"\u043f\u0440\u043e\u0446\u0435\u043d\u0442":3,"\u043f\u0443\u0441\u0442":2,"\u043f\u0443\u0442":[2,3],"\u0440\u0430\u0431\u043e\u0442":[2,3],"\u0440\u0430\u0432\u043d":2,"\u0440\u0430\u0437\u043c\u0435\u0440":[2,3],"\u0440\u0430\u0437\u0440\u0435\u0437":3,"\u0440\u0430\u043d\u0436\u0438\u0440\u043e\u0432\u0430\u043d":3,"\u0440\u0435\u0433\u0443\u043b\u044f\u0440\u0438\u0437\u0430\u0442\u043e\u0440":[],"\u0440\u0435\u0437\u0443\u043b\u044c\u0442\u0430\u0442":[1,2,3],"\u0440\u0435\u043a\u0432\u0435\u0441\u0442":1,"\u0440\u0435\u043a\u0443\u0440\u0441\u0438\u0432\u043d":3,"\u0440\u0443\u0431\u0440\u0438\u043a":[2,3],"\u0441":[2,3],"\u0441\u0431\u0430\u043b\u0430\u043d\u0441\u0438\u0440\u043e\u0432\u0430":2,"\u0441\u0431\u0430\u043b\u0430\u043d\u0441\u0438\u0440\u043e\u0432\u0430\u043d":2,"\u0441\u0435\u0441\u0441":2,"\u0441\u043a\u043e\u0440":2,"\u0441\u043b\u043e\u0432\u0430\u0440":[2,3],"\u0441\u043e\u0434\u0435\u0440\u0436":[2,3],"\u0441\u043e\u0434\u0435\u0440\u0436\u0430":[2,3],"\u0441\u043e\u0437\u0434\u0430":[2,3],"\u0441\u043e\u0437\u0434\u0430\u0432\u0430":3,"\u0441\u043e\u043e\u0442\u0432\u0435\u0442\u0441\u0442\u0432":3,"\u0441\u043e\u0441\u0442\u0430":0,"\u0441\u043e\u0445\u0440\u0430\u043d\u0435\u043d":[2,3],"\u0441\u043e\u0445\u0440\u0430\u043d\u044f":[2,3],"\u0441\u043f\u0438\u0441\u043e\u043a":[2,3],"\u0441\u043f\u043e\u0441\u043e\u0431":3,"\u0441\u0440\u0435\u0434\u043d":[2,3],"\u0441\u0442\u0430\u0442\u0443\u0441":2,"\u0441\u0442\u0440\u043e\u043a":3,"\u0441\u0443\u0449\u0435\u0441\u0442\u0432":3,"\u0441\u0447\u0438\u0442\u0430":3,"\u0441\u044b\u0440":3,"\u0441\u044d\u043c\u043f\u043b\u0438\u0440":2,"\u0442\u0430\u043a":[2,3],"\u0442\u0435\u043a\u0441\u0442":3,"\u0442\u0435\u043a\u0441\u0442\u043e\u0432":3,"\u0442\u0435\u043a\u0443\u0449":2,"\u0442\u0435\u043c":3,"\u0442\u0435\u043c\u0430\u0442\u0438\u0447\u0435\u043a":3,"\u0442\u0435\u043c\u0430\u0442\u0438\u0447\u0435\u0441\u043a":[2,3],"\u0442\u0435\u0441\u0442\u0438\u0440\u0443\u0435\u043c":3,"\u0442\u0435\u0441\u0442\u043e\u0432":3,"\u0442\u0438\u043f":[1,2,3],"\u0442\u043e":2,"\u0442\u043e\u043a":3,"\u0442\u043e\u043a\u0435\u043d":3,"\u0442\u043e\u043b\u044c\u043a":2,"\u0442\u0440\u0435\u043d\u0438\u0440\u043e\u0432\u043e\u0447\u043d":[2,3],"\u0442\u044d\u0442":3,"\u0443\u0434\u0430\u043b":3,"\u0443\u0434\u0430\u043b\u0435\u043d":[],"\u0443\u0434\u0430\u043b\u044f":3,"\u0443\u0434\u043a":3,"\u0443\u043a\u0430\u0437\u0430\u0442\u0435\u043b":0,"\u0443\u043c\u043e\u043b\u0447\u0430\u043d":3,"\u0443\u0447\u0430\u0441\u0442":2,"\u0443\u0447\u0430\u0441\u0442\u0432\u043e\u0432\u0430":2,"\u0443\u0447\u0438\u0442\u044b\u0432\u0430":2,"\u0444\u0430\u0439\u043b":[2,3],"\u0444\u043e\u043d\u043e\u0432":3,"\u0444\u043e\u0440\u043c\u0430\u0442":[2,3],"\u0444\u0443\u043d\u043a\u0446":[2,3],"\u0445\u0430\u0440\u0430\u043a\u0442\u0435\u0440\u0438\u0441\u0442\u0438\u043a":[],"\u0445\u0440\u0430\u043d":[2,3],"\u0445\u0440\u0430\u043d\u0435\u043d":3,"\u0445\u0440\u0430\u043d\u044f":2,"\u0446\u0435\u043d\u0442\u0440\u043e\u0438\u0434":3,"\u0447\u0430\u0441\u0442\u043e\u0442":3,"\u0447\u0435\u0440\u0435\u0437":3,"\u0447\u0442\u043e":3,"\u0447\u0442\u043e\u0431":2,"\u044d\u043a\u0441\u043f\u0435\u0440\u0438\u043c\u0435\u043d\u0442":3,"\u044d\u043f\u043e\u0445":2,"\u044d\u0442":[2,3],"\u044f\u0437\u044b\u043a":3,"class":[1,2,3],"default":[],"do":[],"false":3,"float":3,"for":[2,3],"function":3,"in":3,"int":[2,3],"json\u044a":3,"static":2,"true":3,"with":3,_config:2,_config_path:2,_docs_of_rubrics:2,_path_batches_wiki:2,a:[],adddocumentstomodel:2,adddocumentstomodelrequest:2,adddocumentstomodelresponse:2,analogy:3,and:2,ap:0,args:[],artm:[2,3],artm_model:1,artm_model_params:3,as:[],at:2,average_rubric_size:2,backgroung:[],batch_names:3,batch_vectorizer:2,batchvectorizer:[2,3],bcg_topic_list:3,be:[],bool:3,bow:3,bpe:3,bpe_models:1,bpe_path:3,by:[],calculate:[],calculate_search_quality:3,callable:3,class_ids:2,classes:3,cls_ids:3,config:2,config_experiment:3,contains:[],context:[1,2],convert_doc:3,cos:[],cos_distribution:[],cosine:3,count:3,counter:3,created:[],current_languages:3,data:[],data_dir:2,data_manager:2,dataframe:3,dict:[2,3],dictionary:3,different:[],distribution:3,doc:3,doc_id:[2,3],docid:3,docs:2,docs_from_pack:3,document:[],documentpack:3,documents:[],dump_train_centroids:3,emb_metrics:3,ensure_directory:3,enum_type_wrapper:2,enumtypewrapper:2,eucl:3,evaluate:3,evaluation:3,example:[],exception:2,existing:[],expected:[],experiment_config:2,file:[],files:[],folder:[],format:[],from:2,full:2,general:3,generate:[],generate_batches_balanced_by_rubric:2,generate_model_name:2,generate_theta:3,generator:3,get:[2,3],get_analogy_distribution:3,get_cos_distribution:3,get_mean_classes_intersection:3,get_modalities:3,get_modality_distribution:2,get_num_entries:3,get_rubric_of_train_docs:[],get_subsamples:3,get_topic_profile:3,getdocumentsembedding:1,getdocumentsembeddingrequest:1,getdocumentsembeddingresponse:1,google:2,graphics_for_emb_metrics:[],grnti:[],id:3,id_to_str:3,ids:[],indices:[],inference:0,info:[],init:[],internal:2,intersection:3,is:[],it:2,iterable:3,json:3,keys:[],kind:[],kwargs:3,lang:[],languages:3,latest:2,limit_classwise:3,linalg:3,list:3,load:[],load_bpe_models:3,load_train_data:2,makesubsamples:3,matrices:[],matrix:[],matrix_norm_metric:3,max_dictionary_size:3,means:[],measure:3,measures:[],metrics:[],metrics_to_calculate:3,modality_distribution_all:2,mode:3,model:2,model_info:[],model_main_info:[],model_name:[],model_path:3,model_scores:2,model_scores_value:2,modeldatamanager:2,models:3,models_dir:[],modeltrainer:2,n_bins:[],names:[],none:3,norm:3,not:[],notranslationexception:2,np:3,num_bcg_topic:3,number:[],numer:[],object:2,of:3,or:3,out_file:3,pack:3,pair_analogy:3,pair_cos:3,path:3,path_categories:3,path_experiment:3,path_experiment_result:3,path_model:3,path_models:3,path_results:3,path_rubrics:3,path_save_figs:[],path_subsamples:3,path_test:3,path_to_data:3,path_to_save_subsamples:3,path_train_centroids:3,path_train_lang:3,path_train_thetas:3,pathlib:3,pd:3,profiles:3,property:2,protobuf:2,proximity:[],quality:3,quality_experiment:3,quality_of_models:3,rank_metric:3,ranking:[],rebuild:[],recalculate_test_thetas:3,recalculate_train_centroids:3,recursively_unlink:3,request:[1,2],res:3,results:[],returns:2,rubric:[],rubric_dir:1,rubrics:[],rubrics_train:2,save:[],save_docs:3,save_path:3,saving:[],scores:2,scores_value:2,scratch:2,search_quality:3,self:2,server:[1,2],set_metrics:2,show_analogy_distribution:[],show_cos_distribution:[],size:[],starts_from:3,starttraintopicmodel:2,starttraintopicmodelrequest:2,starttraintopicmodelresponse:2,str:[2,3],subsample:[],subsample_size:3,subsamples:[],subsamples_creator:3,subsemples:[],target_file:3,test:3,textio:3,the:2,theta:[],thetas:[],tmp_dir:3,to:[2,3],todo:[],topic:3,topic_0:3,topic_model:[1,2,3],topicmodelbase_pb2:3,topicmodelinference_pb2:1,topicmodelinference_pb2_grpc:1,topicmodelinferenceserviceimpl:1,topicmodelinferenceserviceservicer:1,topicmodeltrain_pb2:2,topicmodeltrain_pb2_grpc:2,topicmodeltrainserviceimpl:2,topicmodeltrainserviceservicer:2,topics:[],train:0,train_conf:2,train_docs:2,train_grnti:[],train_model:2,train_type:2,trainer:2,traintopicmodelstatus:2,traintopicmodelstatusrequest:2,traintopicmodelstatusresponse:2,traintype:2,txt:3,union:3,update:2,update_config:2,update_ds_metrics:2,updatemodelconfiguration:2,updatemodelconfigurationrequest:2,updatemodelconfigurationresponse:2,use:[],use_counters:3,utils:0,v1:[1,2,3],val:3,value:[],vectors:[],visualize:[],vowpal:[2,3],vowpal_wabbit:3,vowpalwabbit:3,vowpalwabbitbpe:2,vw:[2,3],wabbit:[2,3],way:[],where:[],which:[],wiki:3,wikipedia:2,will:[],work_dir:1,write_new_docs:2},titles:["Welcome to Text categorization\u2019s documentation!","ap.inference","ap.train","ap.utils"],titleterms:{and:0,ap:[1,2,3],categorization:0,contents:0,documentation:0,indices:0,inference:1,s:0,tables:0,text:0,to:0,train:2,utils:3,welcome:0}})