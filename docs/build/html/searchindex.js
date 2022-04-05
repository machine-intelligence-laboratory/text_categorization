Search.setIndex({docnames:["index"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":5,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,sphinx:56},filenames:["index.rst"],objects:{"ap.inference.server":[[0,0,1,"","TopicModelInferenceServiceImpl"],[0,2,1,"","serve"]],"ap.inference.server.TopicModelInferenceServiceImpl":[[0,1,1,"","GetDocumentsEmbedding"],[0,1,1,"","get_rubric_of_train_docs"]],"ap.train":[[0,3,0,"-","server"],[0,3,0,"-","trainer"]],"ap.train.data_manager":[[0,0,1,"","ModelDataManager"]],"ap.train.data_manager.ModelDataManager":[[0,4,1,"","class_ids"],[0,4,1,"","dictionary"],[0,1,1,"","generate_batches_balanced_by_rubric"],[0,1,1,"","write_new_docs"]],"ap.train.server":[[0,0,1,"","TopicModelTrainServiceImpl"]],"ap.train.server.TopicModelTrainServiceImpl":[[0,1,1,"","AddDocumentsToModel"],[0,1,1,"","StartTrainTopicModel"],[0,1,1,"","TrainTopicModelStatus"]],"ap.utils":[[0,3,0,"-","bpe"],[0,3,0,"-","config"],[0,3,0,"-","dictionary"],[0,3,0,"-","emb_metrics"],[0,3,0,"-","general"],[0,3,0,"-","graphics_for_emb_metrics"],[0,3,0,"-","rank_metric"],[0,3,0,"-","search_quality"],[0,3,0,"-","subsamples_creator"],[0,3,0,"-","vowpal_wabbit"],[0,3,0,"-","vowpal_wabbit_bpe"]],"ap.utils.bpe":[[0,2,1,"","load_bpe_models"]],"ap.utils.dictionary":[[0,2,1,"","get_num_entries"],[0,2,1,"","limit_classwise"]],"ap.utils.emb_metrics":[[0,2,1,"","generate_theta"],[0,2,1,"","get_analogy_distribution"],[0,2,1,"","get_cos_distribution"],[0,2,1,"","get_mean_classes_intersection"],[0,2,1,"","get_topic_profile"]],"ap.utils.general":[[0,2,1,"","batch_names"],[0,2,1,"","docs_from_pack"],[0,2,1,"","ensure_directory"],[0,2,1,"","id_to_str"],[0,2,1,"","recursively_unlink"]],"ap.utils.graphics_for_emb_metrics":[[0,2,1,"","show_analogy_distribution"],[0,2,1,"","show_cos_distribution"]],"ap.utils.rank_metric":[[0,2,1,"","quality_of_models"]],"ap.utils.subsamples_creator":[[0,0,1,"","MakeSubsamples"]],"ap.utils.subsamples_creator.MakeSubsamples":[[0,1,1,"","get_subsamples"]],ap:[[0,3,0,"-","train"],[0,3,0,"-","utils"]]},objnames:{"0":["py","class","Python \u043a\u043b\u0430\u0441\u0441"],"1":["py","method","Python \u043c\u0435\u0442\u043e\u0434"],"2":["py","function","Python \u0444\u0443\u043d\u043a\u0446\u0438\u044f"],"3":["py","module","Python \u043c\u043e\u0434\u0443\u043b\u044c"],"4":["py","property","Python property"]},objtypes:{"0":"py:class","1":"py:method","2":"py:function","3":"py:module","4":"py:property"},terms:{"1000":0,"\u0430\u043b\u0444\u0430\u0432\u0438\u0442\u043d":0,"\u0431\u0430\u0442\u0447":0,"\u0431\u043e\u043b":0,"\u0431\u0443\u0434\u0435\u0442":0,"\u0431\u0443\u0434\u0443\u0442":0,"\u0432":0,"\u0432\u0435\u0441":0,"\u0432\u0438\u043a\u0438\u043f\u0435\u0434":0,"\u0432\u043e\u0437\u0432\u0440\u0430\u0437\u0430":0,"\u0432\u043e\u0437\u0432\u0440\u0430\u0449\u0430":0,"\u0432\u0441\u0435\u0433":0,"\u0433\u0435\u043d\u0435\u0440\u0430\u0442\u043e\u0440":0,"\u0433\u0435\u043d\u0435\u0440\u0438\u0440":0,"\u0433\u0440\u043d\u0442\u0438":0,"\u0434\u0430\u043d":0,"\u0434\u0430\u0442\u0430\u0441\u0435\u0442":0,"\u0434\u0438\u0440\u0435\u043a\u0442\u043e\u0440":0,"\u0434\u043b\u044f":0,"\u0434\u043e":0,"\u0434\u043e\u0431\u0430\u0432\u043b\u044f":0,"\u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442":0,"\u0435":0,"\u0435\u043c\u0431\u0435\u0434\u0434\u0438\u043d\u0433":0,"\u0435\u0441\u043b":0,"\u0437\u0430\u0433\u0440\u0443\u0436\u0430":0,"\u0437\u0430\u043f\u0440\u043e\u0441":0,"\u0437\u0430\u043f\u0443\u0441\u043a":0,"\u0437\u0430\u043f\u0443\u0441\u043a\u0430":0,"\u0437\u043d\u0430\u0447\u0435\u043d":0,"\u0438":0,"\u0438\u0437":0,"\u0438\u043c":0,"\u0438\u043c\u0435\u043d":0,"\u0438\u043d\u0430\u0447":0,"\u0438\u043d\u0438\u0446\u0438\u0430\u043b\u0438\u0437\u0438\u0440\u043e\u0432\u0430":0,"\u0438\u043d\u0444\u0435\u0440\u0435\u043d\u0441":0,"\u0438\u0441\u043f\u043e\u043b\u044c\u0437":0,"\u0438\u0441\u043f\u043e\u043b\u044c\u0437\u043e\u0432\u0430":0,"\u0438\u0441\u0445\u043e\u0434\u043d":0,"\u043a":0,"\u043a\u0430\u0436\u0434":0,"\u043a\u0430\u0436\u0434\u043e":0,"\u043a\u043b\u0430\u0441\u0441":0,"\u043a\u043e\u043b\u0438\u0447\u0435\u0441\u0442\u0432":0,"\u043a\u043e\u043d\u0432\u0435\u0440\u0442\u0438\u0440":0,"\u043a\u043e\u043d\u0442\u0435\u043a\u0441\u0442":0,"\u043a\u043e\u043d\u0444\u0438\u0433":0,"\u043a\u043e\u0442\u043e\u0440":0,"\u043c\u0430\u043a\u0441\u0438\u043c\u0430\u043b\u044c\u043d":0,"\u043c\u0435\u0442\u0440\u0438\u043a":0,"\u043c\u043e\u0434\u0430\u043b\u044c\u043d":0,"\u043c\u043e\u0434\u0435\u043b":0,"\u043c\u043e\u0434\u0443\u043b":0,"\u043d\u0430":0,"\u043d\u0430\u0437\u0432\u0430\u043d":0,"\u043d\u0435":0,"\u043d\u0435\u0442":0,"\u043e\u0431\u0440\u0430\u0437":0,"\u043e\u0431\u0443\u0447\u0435\u043d":0,"\u043e\u0433\u0440\u0430\u043d\u0438\u0447\u0438\u0432\u0430":0,"\u043e\u0434\u0438\u043d\u0430\u043a\u043e\u0432":0,"\u043e\u0442\u0432\u0435\u0442":0,"\u043e\u0442\u043d\u043e\u0441\u0438\u0442\u0435\u043b\u044c\u043d":0,"\u043f\u0430\u043f\u043a":0,"\u043f\u043e":0,"\u043f\u043e\u0434\u0434\u0435\u0440\u0436\u0430\u043d":0,"\u043f\u043e\u0438\u0441\u043a":0,"\u043f\u043e\u0441\u043b\u0435\u0434\u043d":0,"\u043f\u043e\u0441\u0442\u0440\u043e\u0435\u043d":0,"\u043f\u0440\u0438\u043d\u0438\u043c\u0430":0,"\u043f\u0440\u0438\u0441\u0443\u0442\u0441\u0442\u0432":0,"\u043f\u0440\u043e\u043c\u0435\u0436\u0443\u0442\u043e\u0447\u043d":0,"\u043f\u0443\u0441\u0442":0,"\u043f\u0443\u0442":0,"\u0440\u0430\u0431\u043e\u0442":0,"\u0440\u0430\u0432\u043d":0,"\u0440\u0430\u0437\u043c\u0435\u0440":0,"\u0440\u0430\u0437\u0440\u0435\u0437":0,"\u0440\u0435\u0437\u0443\u043b\u044c\u0442\u0430\u0442":0,"\u0440\u0435\u043a\u0432\u0435\u0441\u0442":0,"\u0440\u0435\u043a\u0443\u0440\u0441\u0438\u0432\u043d":0,"\u0440\u0443\u0431\u0440\u0438\u043a":0,"\u0441":0,"\u0441\u0431\u0430\u043b\u0430\u043d\u0441\u0438\u0440\u043e\u0432\u0430":0,"\u0441\u0431\u0430\u043b\u0430\u043d\u0441\u0438\u0440\u043e\u0432\u0430\u043d":0,"\u0441\u0435\u0440\u0432\u0435\u0440":0,"\u0441\u0435\u0441\u0441":0,"\u0441\u043b\u043e\u0432\u0430\u0440":0,"\u0441\u043e\u0437\u0434\u0430":0,"\u0441\u043e\u043e\u0442\u0432\u0435\u0442\u0441\u0442\u0432":0,"\u0441\u043e\u0441\u0442\u0430":0,"\u0441\u043e\u0445\u0440\u0430\u043d\u044f":0,"\u0441\u0440\u0435\u0434\u043d":0,"\u0441\u0442\u0430\u0442\u0443\u0441":0,"\u0441\u0442\u0440\u043e\u043a":0,"\u0441\u044d\u043c\u043f\u043b\u0438\u0440":0,"\u0442\u0430\u043a":0,"\u0442\u0435\u043a\u0441\u0442\u043e\u0432":0,"\u0442\u0435\u043a\u0443\u0449":0,"\u0442\u0435\u043c\u0430\u0442\u0438\u0447\u0435\u0441\u043a":0,"\u0442\u0438\u043f":0,"\u0442\u043e":0,"\u0442\u043e\u043a\u0435\u043d":0,"\u0442\u043e\u043b\u044c\u043a":0,"\u0442\u0440\u0435\u043d\u0438\u0440\u043e\u0432\u043e\u0447\u043d":0,"\u0443\u0434\u0430\u043b\u0435\u043d":0,"\u0443\u043a\u0430\u0437\u0430\u0442\u0435\u043b":0,"\u0443\u0447\u0430\u0441\u0442":0,"\u0443\u0447\u0430\u0441\u0442\u0432\u043e\u0432\u0430":0,"\u0444\u0430\u0439\u043b":0,"\u0444\u043e\u0440\u043c\u0430\u0442":0,"\u0445\u0440\u0430\u043d":0,"\u0445\u0440\u0430\u043d\u0435\u043d":0,"\u0447\u0442\u043e":0,"\u0447\u0442\u043e\u0431":0,"\u044d\u043f\u043e\u0445":0,"\u044d\u0442":0,"class":0,"default":0,"do":0,"false":0,"for":0,"function":0,"in":0,"int":0,"return":0,"true":0,"with":0,_config:0,_path_batches_wiki:0,a:0,adddocumentstomodel:0,adddocumentstomodelrequest:0,adddocumentstomodelresponse:0,analogy:0,any:0,args:0,artm:0,artm_model:0,as:0,average_rubric_size:0,backgroung:0,batch_names:0,batch_vectorizer:0,batches_utils:0,batchvectorizer:0,bcg_topic_list:0,be:0,bool:0,bpe:0,bpe_models:0,bpe_path:0,by:0,calculate:0,callable:0,class_ids:0,classes:0,cls_ids:0,contains:0,context:0,cos:0,cos_distribution:0,cosine:0,count:0,created:0,current_languages:0,data:0,data_dir:0,data_manager:0,dataframe:0,dict:0,dictionary:0,different:0,distribution:0,docid:0,docs:0,docs_from_pack:0,document:0,documentpack:0,documents:0,emb_metrics:0,ensure_directory:0,eucl:0,evaluate:0,evaluation:0,example:0,existing:0,expected:0,experiment_config:0,file:0,files:0,folder:0,format:0,general:0,generate:0,generate_batches_balanced_by_rubric:0,generate_theta:0,generator:0,get:0,get_analogy_distribution:0,get_cos_distribution:0,get_mean_classes_intersection:0,get_num_entries:0,get_rubric_of_train_docs:0,get_subsamples:0,get_topic_profile:0,getdocumentsembedding:0,getdocumentsembeddingrequest:0,getdocumentsembeddingresponse:0,graphics_for_emb_metrics:0,grnti:0,id:0,id_to_str:0,ids:0,init:0,intersection:0,is:0,iterable:0,json:0,keys:0,kind:0,kwargs:0,lang:0,languages:0,limit_classwise:0,linalg:0,list:0,load:0,load_bpe_models:0,makesubsamples:0,matrices:0,matrix:0,matrix_norm_metric:0,max_dictionary_size:0,means:0,measure:0,measures:0,metrics:0,metrics_to_calculate:0,mode:0,model:0,model_name:0,modeldatamanager:0,models:0,models_dir:0,n_bins:0,names:0,none:0,norm:0,not:0,np:0,number:0,numer:0,of:0,or:0,out_file:0,pack:0,pair_analogy:0,pair_cos:0,param:0,path:0,path_categories:0,path_experiment_result:0,path_model:0,path_models:0,path_rubrics:0,path_save_figs:0,path_subsamples:0,path_test:0,path_to_data:0,path_to_save_subsamples:0,path_train_lang:0,pathlib:0,pd:0,profiles:0,property:0,proximity:0,quality:0,quality_experiment:0,quality_of_models:0,rank_metric:0,ranking:0,rebuild:0,recalculate_test_thetas:0,recursively_unlink:0,request:0,results:0,returns:0,rubric:0,rubric_dir:0,rubrics:0,save:0,save_path:0,saving:0,self:0,serve:0,server:0,show_analogy_distribution:0,show_cos_distribution:0,size:0,starts_from:0,starttraintopicmodel:0,starttraintopicmodelrequest:0,starttraintopicmodelresponse:0,str:0,subsample:0,subsample_size:0,subsamples:0,subsamples_creator:0,subsemples:0,test:0,the:0,theta:0,thetas:0,tmp_dir:0,todo:0,topic:0,topic_0:0,topic_model:0,topicmodelbase_pb2:0,topicmodelinference_pb2:0,topicmodelinferenceserviceimpl:0,topicmodeltrain_pb2:0,topicmodeltrainserviceimpl:0,topics:0,train_conf:0,train_grnti:0,traintopicmodelstatus:0,traintopicmodelstatusrequest:0,traintopicmodelstatusresponse:0,txt:0,typing:0,use:0,utils:0,v1:0,value:0,vectors:0,visualize:0,vw_writer:0,way:0,where:0,which:0,wiki:0,will:0,work_dir:0,write_new_docs:0},titles:["Welcome to Text categorization\u2019s documentation!"],titleterms:{and:0,ap:0,categorization:0,documentation:0,indices:0,inference:0,s:0,tables:0,text:0,to:0,train:0,utils1:0,welcome:0}})