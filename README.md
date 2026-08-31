{
  "meta": {
    "n_mc": 20,
    "seed": 12345,
    "modeles": {
      "A (etroit)": [
        "archi1",
        "archi2"
      ],
      "B (randomise)": [
        "archi1",
        "archi2"
      ]
    },
    "facteur_divergence": 100.0
  },
  "axes": {
    "commandes": {
      "titre": "Familles de commandes jamais vues a l'entrainement",
      "xlabel": "",
      "groupe": "commandes",
      "scenarios": {
        "commandes_ref": {
          "libelle": "3 phases (reference in-distrib.)",
          "x": null,
          "T": 160,
          "nominal": true,
          "reference": true,
          "ekf_d2": 6.51749067902565,
          "ekf_all": 7.942582079768181,
          "modeles": {
            "A (etroit)": {
              "archi1": {
                "mse_d2": 83.31402845531701,
                "mse_all": 39.69853204190731,
                "div_rate": 0.05,
                "delta_d2": 11.066377155377126,
                "delta_all": 6.988127362912282,
                "delta_transitoire": 1.378663249547375,
                "delta_etabli": 12.711056568187606,
                "delta_bloc_debut": 11.081312032810914,
                "delta_bloc_fin": 11.081312032810914
              },
              "archi2": {
                "mse_d2": 12.709326848387718,
                "mse_all": 8.211090767383576,
                "div_rate": 0.0,
                "delta_d2": 2.900421299480663,
                "delta_all": 0.14439141369514258,
                "delta_transitoire": 1.0075769590410588,
                "delta_etabli": 3.9388485117059844,
                "delta_bloc_debut": 2.908321370193488,
                "delta_bloc_fin": 2.908321370193488
              }
            },
            "B (randomise)": {
              "archi1": {
                "mse_d2": 23.871446093916894,
                "mse_all": 14.803427636623383,
                "div_rate": 0.05,
                "delta_d2": 5.637983099312555,
                "delta_all": 2.7040057349567674,
                "delta_transitoire": 1.3558497029105019,
                "delta_etabli": 7.048046392589782,
                "delta_bloc_debut": 5.649766285481343,
                "delta_bloc_fin": 5.649766285481343
              },
              "archi2": {
                "mse_d2": 18.782190701365472,
                "mse_all": 12.45997794866562,
                "div_rate": 0.0,
                "delta_d2": 4.5966582702348004,
                "delta_all": 1.9555556213805536,
                "delta_transitoire": 2.172352370047407,
                "delta_etabli": 5.585608313548938,
                "delta_bloc_debut": 4.607243617181656,
                "delta_bloc_fin": 4.607243617181656
              }
            }
          }
        },
        "commandes_creneaux": {
          "libelle": "Creneaux (bang-bang)",
          "x": null,
          "T": 160,
          "nominal": false,
          "reference": false,
          "ekf_d2": 9.858406576514245,
          "ekf_all": 6.721204957365989,
          "modeles": {
            "A (etroit)": {
              "archi1": {
                "mse_d2": 1518.2817955613136,
                "mse_all": 1279.4668178200723,
                "div_rate": 0.3,
                "delta_d2": 21.875456596104716,
                "delta_all": 22.79581888171187,
                "delta_transitoire": 1.4511288823637747,
                "delta_etabli": 25.49042171042629,
                "delta_bloc_debut": 21.88351579330556,
                "delta_bloc_fin": 21.88351579330556
              },
              "archi2": {
                "mse_d2": 536.2380319535732,
                "mse_all": 419.02170461416245,
                "div_rate": 0.1,
                "delta_d2": 17.355508872953237,
                "delta_all": 17.947893802275708,
                "delta_transitoire": 0.3132191371557718,
                "delta_etabli": 21.581369349663323,
                "delta_bloc_debut": 17.363472020445702,
                "delta_bloc_fin": 17.363472020445702
              }
            },
            "B (randomise)": {
              "archi1": {
                "mse_d2": 34.67862172424793,
                "mse_all": 19.088622030615806,
                "div_rate": 0.05,
                "delta_d2": 5.46255103249732,
                "delta_all": 4.533274396370551,
                "delta_transitoire": 0.7824145869490009,
                "delta_etabli": 7.2354441770760385,
                "delta_bloc_debut": 5.4683583705710115,
                "delta_bloc_fin": 5.4683583705710115
              },
              "archi2": {
                "mse_d2": 17.55591170489788,
                "mse_all": 13.347554272413253,
                "div_rate": 0.0,
                "delta_d2": 2.506166627628889,
                "delta_all": 2.9795455637631116,
                "delta_transitoire": 1.0886639985906497,
                "delta_etabli": 3.588599768038626,
                "delta_bloc_debut": 2.5097250743701247,
                "delta_bloc_fin": 2.5097250743701247
              }
            }
          }
        },
        "commandes_chirp": {
          "libelle": "Chirp",
          "x": null,
          "T": 160,
          "nominal": false,
          "reference": false,
          "ekf_d2": 11.74271875321865,
          "ekf_all": 8.173813955485821,
          "modeles": {
            "A (etroit)": {
              "archi1": {
                "mse_d2": 476.7661349773407,
                "mse_all": 217.2410272359848,
                "div_rate": 0.05,
                "delta_d2": 16.085367402627583,
                "delta_all": 14.245170992168866,
                "delta_transitoire": 0.6945056490427952,
                "delta_etabli": 19.425904989731432,
                "delta_bloc_debut": 16.095043184777403,
                "delta_bloc_fin": 16.095043184777403
              },
              "archi2": {
                "mse_d2": 10.83267293870449,
                "mse_all": 9.367965787649155,
                "div_rate": 0.0,
                "delta_d2": -0.3503302825673233,
                "delta_all": 0.5922054750514931,
                "delta_transitoire": 0.3810467982899086,
                "delta_etabli": -0.23807712714042698,
                "delta_bloc_debut": -0.3511647809806346,
                "delta_bloc_fin": -0.3511647809806346
              }
            },
            "B (randomise)": {
              "archi1": {
                "mse_d2": 22.377297124266626,
                "mse_all": 15.020185112953186,
                "div_rate": 0.0,
                "delta_d2": 2.800389691105093,
                "delta_all": 2.642505365739491,
                "delta_transitoire": 0.11546930762010885,
                "delta_etabli": 4.282118869210104,
                "delta_bloc_debut": 2.8051067235095246,
                "delta_bloc_fin": 2.8051067235095246
              },
              "archi2": {
                "mse_d2": 6.407949009537697,
                "mse_all": 5.8950498953461645,
                "div_rate": 0.0,
                "delta_d2": -2.6304961208742066,
                "delta_all": -1.419372632046603,
                "delta_transitoire": 0.9295608224973058,
                "delta_etabli": -2.347896008936562,
                "delta_bloc_debut": -2.638771547621751,
                "delta_bloc_fin": -2.638771547621751
              }
            }
          }
        },
        "commandes_virage": {
          "libelle": "Virage coordonne",
          "x": null,
          "T": 160,
          "nominal": false,
          "reference": false,
          "ekf_d2": 5.8434849113225935,
          "ekf_all": 6.330306500196457,
          "modeles": {
            "A (etroit)": {
              "archi1": {
                "mse_d2": 3013.9981404781342,
                "mse_all": 2277.069451713562,
                "div_rate": 0.15,
                "delta_d2": 27.124710530657993,
                "delta_all": 25.559515388511407,
                "delta_transitoire": 1.3546068665332167,
                "delta_etabli": 31.380070077337017,
                "delta_bloc_debut": 27.147248948482748,
                "delta_bloc_fin": 27.147248948482748
              },
              "archi2": {
                "mse_d2": 9093244.203187222,
                "mse_all": 11553309.372165013,
                "div_rate": 0.1,
                "delta_d2": 61.920469274489314,
                "delta_all": 62.61281664790665,
                "delta_transitoire": 1.1332916207214285,
                "delta_etabli": 67.09575701356738,
                "delta_bloc_debut": 61.943051563809775,
                "delta_bloc_fin": 61.943051563809775
              }
            },
            "B (randomise)": {
              "archi1": {
                "mse_d2": 17.011119878292085,
                "mse_all": 9.465187752246857,
                "div_rate": 0.0,
                "delta_d2": 4.640609781156588,
                "delta_all": 1.7470449492989892,
                "delta_transitoire": 0.6351352338226273,
                "delta_etabli": 5.366162400440243,
                "delta_bloc_debut": 4.655448152190002,
                "delta_bloc_fin": 4.655448152190002
              },
              "archi2": {
                "mse_d2": 7.182824686169624,
                "mse_all": 5.519546943902969,
                "div_rate": 0.0,
                "delta_d2": 0.8962333960855352,
                "delta_all": -0.5952130677486447,
                "delta_transitoire": 2.0780571822907064,
                "delta_etabli": 2.1714239191886966,
                "delta_bloc_debut": 0.9004531243202274,
                "delta_bloc_fin": 0.9004531243202274
              }
            }
          }
        },
        "commandes_stopgo": {
          "libelle": "Stop-and-go",
          "x": null,
          "T": 160,
          "nominal": false,
          "reference": false,
          "ekf_d2": 7.149638153612614,
          "ekf_all": 5.631685265898705,
          "modeles": {
            "A (etroit)": {
              "archi1": {
                "mse_d2": 200.86123192608358,
                "mse_all": 112.1912140622735,
                "div_rate": 0.15,
                "delta_d2": 14.486120594690547,
                "delta_all": 12.993204720105833,
                "delta_transitoire": 0.5589544307114511,
                "delta_etabli": 17.903443089377816,
                "delta_bloc_debut": 14.499763925414586,
                "delta_bloc_fin": 14.499763925414586
              },
              "archi2": {
                "mse_d2": 27.759731811285018,
                "mse_all": 15.460382598638535,
                "div_rate": 0.05,
                "delta_d2": 5.891312035345194,
                "delta_all": 4.3858186151953875,
                "delta_transitoire": 0.40647554298984745,
                "delta_etabli": 8.573474640610515,
                "delta_bloc_debut": 5.901819122377587,
                "delta_bloc_fin": 5.901819122377587
              }
            },
            "B (randomise)": {
              "archi1": {
                "mse_d2": 7.952614513039589,
                "mse_all": 6.234278473258018,
                "div_rate": 0.0,
                "delta_d2": 0.46225868894112454,
                "delta_all": 0.4414782177858501,
                "delta_transitoire": 0.0802065303107276,
                "delta_etabli": 1.4722805695153731,
                "delta_bloc_debut": 0.46368893347658113,
                "delta_bloc_fin": 0.46368893347658113
              },
              "archi2": {
                "mse_d2": 12.990627613663673,
                "mse_all": 7.099591276049614,
                "div_rate": 0.0,
                "delta_d2": 2.593460710455438,
                "delta_all": 1.0059497137925988,
                "delta_transitoire": 0.91020418870418,
                "delta_etabli": 4.005290905717942,
                "delta_bloc_debut": 2.599827008834695,
                "delta_bloc_fin": 2.599827008834695
              }
            }
          }
        }
      }
    },
    "amplitude": {
      "titre": "Extrapolation en amplitude de commande (famille creneaux)",
      "xlabel": "Amplitude de commande A",
      "groupe": "commandes",
      "scenarios": {
        "amplitude_0.5": {
          "libelle": "A = 0.5",
          "x": 0.5,
          "T": 160,
          "nominal": false,
          "reference": false,
          "ekf_d2": 9.04306703209877,
          "ekf_all": 8.5881556391716,
          "modeles": {
            "A (etroit)": {
              "archi1": {
                "mse_d2": 30.1580549120903,
                "mse_all": 22.387757325172423,
                "div_rate": 0.0,
                "delta_d2": 5.230875775639975,
                "delta_all": 4.161106843186601,
                "delta_transitoire": 1.635241248375572,
                "delta_etabli": 6.228200300164871,
                "delta_bloc_debut": 5.238063134685782,
                "delta_bloc_fin": 5.238063134685782
              },
              "archi2": {
                "mse_d2": 14.678726284205913,
                "mse_all": 10.978480032086372,
                "div_rate": 0.0,
                "delta_d2": 2.1037262214053105,
                "delta_all": 1.0664230995621402,
                "delta_transitoire": 1.3982397385773906,
                "delta_etabli": 3.1828034771253,
                "delta_bloc_debut": 2.1076688477093386,
                "delta_bloc_fin": 2.1076688477093386
              }
            },
            "B (randomise)": {
              "archi1": {
                "mse_d2": 20.145811635255814,
                "mse_all": 13.77721972167492,
                "div_rate": 0.0,
                "delta_d2": 3.4786901887165116,
                "delta_all": 2.0526167831818576,
                "delta_transitoire": 1.2656283603438736,
                "delta_etabli": 3.9993131029425246,
                "delta_bloc_debut": 3.4843487301231644,
                "delta_bloc_fin": 3.4843487301231644
              },
              "archi2": {
                "mse_d2": 11.01793563514948,
                "mse_all": 9.352901756763458,
                "div_rate": 0.0,
                "delta_d2": 0.857844808939303,
                "delta_all": 0.3704646617008042,
                "delta_transitoire": 1.4285599729511536,
                "delta_etabli": 1.5180188873433622,
                "delta_bloc_debut": 0.8596859603045424,
                "delta_bloc_fin": 0.8596859603045424
              }
            }
          }
        },
        "amplitude_1": {
          "libelle": "A = 1",
          "x": 1.0,
          "T": 160,
          "nominal": false,
          "reference": true,
          "ekf_d2": 27.938757456839085,
          "ekf_all": 17.66230803281069,
          "modeles": {
            "A (etroit)": {
              "archi1": {
                "mse_d2": 270.89170310497286,
                "mse_all": 143.2679738998413,
                "div_rate": 0.1,
                "delta_d2": 9.865886161523527,
                "delta_all": 9.091016644823613,
                "delta_transitoire": 0.6780877955648225,
                "delta_etabli": 11.896364885887836,
                "delta_bloc_debut": 9.8693361015549,
                "delta_bloc_fin": 9.8693361015549
              },
              "archi2": {
                "mse_d2": 352.86145552098753,
                "mse_all": 583.5671894997358,
                "div_rate": 0.05,
                "delta_d2": 11.013971337393599,
                "delta_all": 15.19043411476392,
                "delta_transitoire": 0.6459290671025888,
                "delta_etabli": 15.139153411031916,
                "delta_bloc_debut": 11.017513525052152,
                "delta_bloc_fin": 11.017513525052152
              }
            },
            "B (randomise)": {
              "archi1": {
                "mse_d2": 34.41129640340805,
                "mse_all": 20.11800488233566,
                "div_rate": 0.0,
                "delta_d2": 0.9049394696661954,
                "delta_all": 0.5653745456454646,
                "delta_transitoire": 0.15426495087752856,
                "delta_etabli": 1.3365554753810571,
                "delta_bloc_debut": 0.9056634908408041,
                "delta_bloc_fin": 0.9056634908408041
              },
              "archi2": {
                "mse_d2": 17.02008988261223,
                "mse_all": 12.326293063163757,
                "div_rate": 0.0,
                "delta_d2": -2.1524523822325956,
                "delta_all": -1.5621496558051295,
                "delta_transitoire": 1.3644896463977907,
                "delta_etabli": -1.8507723993611842,
                "delta_bloc_debut": -2.154921263883561,
                "delta_bloc_fin": -2.154921263883561
              }
            }
          }
        },
        "amplitude_1.5": {
          "libelle": "A = 1.5",
          "x": 1.5,
          "T": 160,
          "nominal": false,
          "reference": false,
          "ekf_d2": 5.210632283985615,
          "ekf_all": 3.679335191845894,
          "modeles": {
            "A (etroit)": {
              "archi1": {
                "mse_d2": 1558.715516754985,
                "mse_all": 926.7685111701488,
                "div_rate": 0.4,
                "delta_d2": 24.758764328115387,
                "delta_all": 24.012019149561716,
                "delta_transitoire": 1.317270659262897,
                "delta_etabli": 29.193389696750486,
                "delta_bloc_debut": 24.78594364538935,
                "delta_bloc_fin": 24.78594364538935
              },
              "archi2": {
                "mse_d2": 19682.491156339645,
                "mse_all": 31611.725639504195,
                "div_rate": 0.2,
                "delta_d2": 35.771896390332415,
                "delta_all": 39.34078849628474,
                "delta_transitoire": 1.068018425477871,
                "delta_etabli": 41.47892268423643,
                "delta_bloc_debut": 35.799159033289556,
                "delta_bloc_fin": 35.799159033289556
              }
            },
            "B (randomise)": {
              "archi1": {
                "mse_d2": 24.377192172408105,
                "mse_all": 11.493500131368638,
                "div_rate": 0.0,
                "delta_d2": 6.70093255064049,
                "delta_all": 4.946829507438027,
                "delta_transitoire": -0.5534067210196412,
                "delta_etabli": 9.13452513045003,
                "delta_bloc_debut": 6.7223880944609204,
                "delta_bloc_fin": 6.7223880944609204
              },
              "archi2": {
                "mse_d2": 6.3658747464418415,
                "mse_all": 4.385120482742787,
                "div_rate": 0.0,
                "delta_d2": 0.8696766334052766,
                "delta_all": 0.7621217587986505,
                "delta_transitoire": 1.3191454852081268,
                "delta_etabli": 2.305560263499657,
                "delta_bloc_debut": 0.874638095473788,
                "delta_bloc_fin": 0.874638095473788
              }
            }
          }
        },
        "amplitude_2": {
          "libelle": "A = 2",
          "x": 2.0,
          "T": 160,
          "nominal": false,
          "reference": false,
          "ekf_d2": 12.103396737575531,
          "ekf_all": 7.995363402366638,
          "modeles": {
            "A (etroit)": {
              "archi1": {
                "mse_d2": 1253.2655858516694,
                "mse_all": 717.1150191545487,
                "div_rate": 0.4,
                "delta_d2": 20.15135845016008,
                "delta_all": 19.527506104719432,
                "delta_transitoire": 1.5584941289698815,
                "delta_etabli": 23.181131659534074,
                "delta_bloc_debut": 20.159438722205664,
                "delta_bloc_fin": 20.159438722205664
              },
              "archi2": {
                "mse_d2": 21571.300719988347,
                "mse_all": 26896.260951697826,
                "div_rate": 0.25,
                "delta_d2": 32.50969063984246,
                "delta_all": 35.268537017816655,
                "delta_transitoire": 0.6834979548618413,
                "delta_etabli": 37.21819959180929,
                "delta_bloc_debut": 32.51784426392153,
                "delta_bloc_fin": 32.51784426392153
              }
            },
            "B (randomise)": {
              "archi1": {
                "mse_d2": 12.02203086912632,
                "mse_all": 10.17229684740305,
                "div_rate": 0.0,
                "delta_d2": -0.029294304946376037,
                "delta_all": 1.04580817319472,
                "delta_transitoire": 0.442342708997053,
                "delta_etabli": -0.6445566623198183,
                "delta_bloc_debut": -0.029349689891537994,
                "delta_bloc_fin": -0.029349689891537994
              },
              "archi2": {
                "mse_d2": 8.03637646138668,
                "mse_all": 6.492269062995911,
                "div_rate": 0.0,
                "delta_d2": -1.7784699633877659,
                "delta_all": -0.9044169767308906,
                "delta_transitoire": 0.9646520533964865,
                "delta_etabli": -1.2435860567286892,
                "delta_bloc_debut": -1.7826048090341662,
                "delta_bloc_fin": -1.7826048090341662
              }
            }
          }
        },
        "amplitude_2.5": {
          "libelle": "A = 2.5",
          "x": 2.5,
          "T": 160,
          "nominal": false,
          "reference": false,
          "ekf_d2": 9.701677110791206,
          "ekf_all": 6.883390012383461,
          "modeles": {
            "A (etroit)": {
              "archi1": {
                "mse_d2": 5841.315887355804,
                "mse_all": 4050.8967821121214,
                "div_rate": 0.6,
                "delta_d2": 27.796638762704017,
                "delta_all": 27.69748800044885,
                "delta_transitoire": 2.891985834454976,
                "delta_etabli": 31.27325835586059,
                "delta_bloc_debut": 27.811448853251164,
                "delta_bloc_fin": 27.811448853251164
              },
              "archi2": {
                "mse_d2": 917.0628092288971,
                "mse_all": 924.3928537249565,
                "div_rate": 0.1,
                "delta_d2": 19.755522648953807,
                "delta_all": 21.28054202041229,
                "delta_transitoire": 1.2698761965716425,
                "delta_etabli": 23.861476588882432,
                "delta_bloc_debut": 19.770200069485217,
                "delta_bloc_fin": 19.770200069485217
              }
            },
            "B (randomise)": {
              "archi1": {
                "mse_d2": 23.849934670329095,
                "mse_all": 15.504927417635917,
                "div_rate": 0.0,
                "delta_d2": 3.906403773280485,
                "delta_all": 3.526673601942141,
                "delta_transitoire": 0.1545254672223163,
                "delta_etabli": 3.387166576108968,
                "delta_bloc_debut": 3.915210119341901,
                "delta_bloc_fin": 3.915210119341901
              },
              "archi2": {
                "mse_d2": 16.04665489792824,
                "mse_all": 8.797577542066573,
                "div_rate": 0.0,
                "delta_d2": 2.1853769627305217,
                "delta_all": 1.0656072606141596,
                "delta_transitoire": 1.9182167902607659,
                "delta_etabli": 2.56200422232207,
                "delta_bloc_debut": 2.1912487104356893,
                "delta_bloc_fin": 2.1912487104356893
              }
            }
          }
        },
        "amplitude_3": {
          "libelle": "A = 3",
          "x": 3.0,
          "T": 160,
          "nominal": false,
          "reference": false,
          "ekf_d2": 17.308484959602357,
          "ekf_all": 9.084218326210976,
          "modeles": {
            "A (etroit)": {
              "archi1": {
                "mse_d2": 1456.7546385258436,
                "mse_all": 726.5190756499767,
                "div_rate": 0.4,
                "delta_d2": 19.251273546006512,
                "delta_all": 19.029594583977428,
                "delta_transitoire": 2.205234847737714,
                "delta_etabli": 22.175226504911322,
                "delta_bloc_debut": 19.256642128419944,
                "delta_bloc_fin": 19.256642128419944
              },
              "archi2": {
                "mse_d2": 447489.2313330531,
                "mse_all": 559570.6311225176,
                "div_rate": 0.3,
                "delta_d2": 44.12523533614939,
                "delta_all": 47.89567349296631,
                "delta_transitoire": 1.1151468795694823,
                "delta_etabli": 48.52350458330144,
                "delta_bloc_debut": 44.130668613194814,
                "delta_bloc_fin": 44.130668613194814
              }
            },
            "B (randomise)": {
              "archi1": {
                "mse_d2": 19.34326029419899,
                "mse_all": 11.871279957890511,
                "div_rate": 0.0,
                "delta_d2": 0.482706209521436,
                "delta_all": 1.162099836609681,
                "delta_transitoire": 1.128106945183821,
                "delta_etabli": 1.829834365651302,
                "delta_bloc_debut": 0.48327769964078526,
                "delta_bloc_fin": 0.48327769964078526
              },
              "archi2": {
                "mse_d2": 16.541019216179848,
                "mse_all": 8.350445383787156,
                "div_rate": 0.0,
                "delta_d2": -0.19696788843315705,
                "delta_all": -0.3657792345826568,
                "delta_transitoire": 0.947362938115647,
                "delta_etabli": 0.6215390111103688,
                "delta_bloc_debut": -0.19722061418707412,
                "delta_bloc_fin": -0.19722061418707412
              }
            }
          }
        }
      }
    },
    "horizon": {
      "titre": "Horizon plus long que celui vu a l'entrainement",
      "xlabel": "Horizon T (pas)",
      "groupe": "horizon",
      "scenarios": {
        "horizon_160": {
          "libelle": "T = 160 (entrainement)",
          "x": 160,
          "T": 160,
          "nominal": true,
          "reference": true,
          "ekf_d2": 11.977607232332229,
          "ekf_all": 8.549485954642297,
          "modeles": {
            "A (etroit)": {
              "archi1": {
                "mse_d2": 25.164289185404776,
                "mse_all": 12.599927136301995,
                "div_rate": 0.0,
                "delta_d2": 3.2241459973825948,
                "delta_all": 1.684280304820619,
                "delta_transitoire": 0.4464192451688708,
                "delta_etabli": 4.2074008184778595,
                "delta_bloc_debut": 3.22960194087774,
                "delta_bloc_fin": 3.22960194087774
              },
              "archi2": {
                "mse_d2": 8.830555713176727,
                "mse_all": 6.367677983641625,
                "div_rate": 0.0,
                "delta_d2": -1.3238203277115477,
                "delta_all": -1.2795891035934321,
                "delta_transitoire": 0.2463592554211241,
                "delta_etabli": -0.619006299569203,
                "delta_bloc_debut": -1.327534718250861,
                "delta_bloc_fin": -1.327534718250861
              }
            },
            "B (randomise)": {
              "archi1": {
                "mse_d2": 21.455939477682115,
                "mse_all": 15.198415648937225,
                "div_rate": 0.0,
                "delta_d2": 2.5317746779285653,
                "delta_all": 2.4985831432257593,
                "delta_transitoire": -0.456723149635273,
                "delta_etabli": 2.6449697585188066,
                "delta_bloc_debut": 2.5363747191084167,
                "delta_bloc_fin": 2.5363747191084167
              },
              "archi2": {
                "mse_d2": 9.371011999249458,
                "mse_all": 5.5742207109928135,
                "div_rate": 0.0,
                "delta_d2": -1.0658357369239246,
                "delta_all": -1.8575584258948359,
                "delta_transitoire": 0.8177954593584879,
                "delta_etabli": -0.5858979437539393,
                "delta_bloc_debut": -1.068734652494538,
                "delta_bloc_fin": -1.068734652494538
              }
            }
          }
        },
        "horizon_320": {
          "libelle": "T = 320 (2x)",
          "x": 320,
          "T": 320,
          "nominal": false,
          "reference": false,
          "ekf_d2": 39.95158553421497,
          "ekf_all": 21.976908469200133,
          "modeles": {
            "A (etroit)": {
              "archi1": {
                "mse_d2": 218483.85181265773,
                "mse_all": 205260.63246930242,
                "div_rate": 0.15,
                "delta_d2": 37.37885324123966,
                "delta_all": 39.70339063354809,
                "delta_transitoire": 0.0455945908659461,
                "delta_etabli": 43.448021749978636,
                "delta_bloc_debut": 3.5963502257356583,
                "delta_bloc_fin": 37.72185763504395
              },
              "archi2": {
                "mse_d2": 100.45265772938728,
                "mse_all": 43.08688159584999,
                "div_rate": 0.05,
                "delta_d2": 4.004274120240195,
                "delta_all": 2.9237846384544532,
                "delta_transitoire": -0.06283941671316515,
                "delta_etabli": 4.673078852170764,
                "delta_bloc_debut": 0.6570836621035681,
                "delta_bloc_fin": 4.192077154252319
              }
            },
            "B (randomise)": {
              "archi1": {
                "mse_d2": 766.1896794438362,
                "mse_all": 462.44076774120333,
                "div_rate": 0.15,
                "delta_d2": 12.82802278178129,
                "delta_all": 13.230895141210038,
                "delta_transitoire": 0.2440908848717799,
                "delta_etabli": 16.8562196222098,
                "delta_bloc_debut": 1.9177957406258987,
                "delta_bloc_fin": 13.144355031270718
              },
              "archi2": {
                "mse_d2": 106.0512179851532,
                "mse_all": 50.64394569396973,
                "div_rate": 0.0,
                "delta_d2": 4.23981641127437,
                "delta_all": 3.6256093501079,
                "delta_transitoire": 0.8945838598612599,
                "delta_etabli": 5.434760134815817,
                "delta_bloc_debut": 1.1010413892993076,
                "delta_bloc_fin": 4.419877390046839
              }
            }
          }
        },
        "horizon_480": {
          "libelle": "T = 480 (3x)",
          "x": 480,
          "T": 480,
          "nominal": false,
          "reference": false,
          "ekf_d2": 266.8822020411491,
          "ekf_all": 161.54534714221955,
          "modeles": {
            "A (etroit)": {
              "archi1": {
                "mse_d2": 1.6339268355285683e+18,
                "mse_all": 1.9148069675938115e+18,
                "div_rate": 0.55,
                "delta_d2": 157.86912993331896,
                "delta_all": 160.73830545215242,
                "delta_transitoire": 0.8736737567809383,
                "delta_etabli": 166.45247500365073,
                "delta_bloc_debut": 3.413457214332989,
                "delta_bloc_fin": 159.23178339304127
              },
              "archi2": {
                "mse_d2": 301.1113878726959,
                "mse_all": 244.41435277462006,
                "div_rate": 0.0,
                "delta_d2": 0.5240756828004993,
                "delta_all": 1.7983225154156468,
                "delta_transitoire": 0.3637882791898419,
                "delta_etabli": 3.8434201579920684,
                "delta_bloc_debut": -6.994787032975088,
                "delta_bloc_fin": 1.2957290284180392
              }
            },
            "B (randomise)": {
              "archi1": {
                "mse_d2": 59122160.82613011,
                "mse_all": 37701506.1953517,
                "div_rate": 0.4,
                "delta_d2": 53.45430686157649,
                "delta_all": 53.68064246942386,
                "delta_transitoire": 0.15276715230512852,
                "delta_etabli": 60.7328220294145,
                "delta_bloc_debut": 3.5572021750180394,
                "delta_bloc_fin": 54.816537575110644
              },
              "archi2": {
                "mse_d2": 611832275608.8457,
                "mse_all": 390384665125.2655,
                "div_rate": 0.3,
                "delta_d2": 93.60312770990754,
                "delta_all": 93.83198295706819,
                "delta_transitoire": 1.5718392706081297,
                "delta_etabli": 102.07028688154412,
                "delta_bloc_debut": -3.173506511646687,
                "delta_bloc_fin": 94.96578119107902
              }
            }
          }
        },
        "horizon_960": {
          "libelle": "T = 960 (6x)",
          "x": 960,
          "T": 960,
          "nominal": false,
          "reference": false,
          "ekf_d2": 1199.0553477525712,
          "ekf_all": 634.3302083492279,
          "modeles": {
            "A (etroit)": {
              "archi1": {
                "mse_d2": 8.989456040773645e+33,
                "mse_all": Infinity,
                "div_rate": 0.7,
                "delta_d2": 308.74894182657164,
                "delta_all": null,
                "delta_transitoire": 1.0206582561418247,
                "delta_etabli": 319.9840935904617,
                "delta_bloc_debut": -0.048093425011659044,
                "delta_bloc_fin": 312.02275850802755
              },
              "archi2": {
                "mse_d2": 302350214373388.75,
                "mse_all": 303571597509313.4,
                "div_rate": 0.9,
                "delta_d2": 114.01671050504847,
                "delta_all": 116.79945742006677,
                "delta_transitoire": 0.4468086251515602,
                "delta_etabli": 123.7232345636985,
                "delta_bloc_debut": 0.20051406627265442,
                "delta_bloc_fin": 117.28961831092478
              }
            },
            "B (randomise)": {
              "archi1": {
                "mse_d2": 1.9483184078750376e+19,
                "mse_all": 1.242513542975489e+19,
                "div_rate": 0.7,
                "delta_d2": 162.10820703434482,
                "delta_all": 162.91985737042717,
                "delta_transitoire": -0.17937627325432753,
                "delta_etabli": 172.04035500101298,
                "delta_bloc_debut": 2.5775625348278774,
                "delta_bloc_fin": 165.3816540330257
              },
              "archi2": {
                "mse_d2": 9.574606866163094e+31,
                "mse_all": 7.020605686726279e+31,
                "div_rate": 0.65,
                "delta_d2": 289.022817204914,
                "delta_all": 290.44059187460215,
                "delta_transitoire": 1.5440015212678952,
                "delta_etabli": 300.1498266406619,
                "delta_bloc_debut": -1.1024630960513968,
                "delta_bloc_fin": 292.29663397043487
              }
            }
          }
        }
      }
    },
    "cadence": {
      "titre": "Cadence du GPS et des mesures de distance",
      "xlabel": "ratio_gps (1 mesure tous les N pas)",
      "groupe": "capteurs",
      "scenarios": {
        "cadence_5": {
          "libelle": "ratio_gps = 5 (entrainement)",
          "x": 5,
          "T": 160,
          "nominal": true,
          "reference": true,
          "ekf_d2": 15.319524490833283,
          "ekf_all": 10.97162597179413,
          "modeles": {
            "A (etroit)": {
              "archi1": {
                "mse_d2": 20.6270340770483,
                "mse_all": 12.240366265177727,
                "div_rate": 0.0,
                "delta_d2": 1.2919150080800113,
                "delta_all": 0.47523419411417755,
                "delta_transitoire": -0.38997891421463615,
                "delta_etabli": 2.5882453444434432,
                "delta_bloc_debut": 1.293203469067615,
                "delta_bloc_fin": 1.293203469067615
              },
              "archi2": {
                "mse_d2": 13.152008117735386,
                "mse_all": 8.628819543123246,
                "div_rate": 0.0,
                "delta_d2": -0.6625321700358102,
                "delta_all": -1.043196073050097,
                "delta_transitoire": -0.4068205846846754,
                "delta_etabli": 0.31311954785753154,
                "delta_bloc_debut": -0.6633574683409442,
                "delta_bloc_fin": -0.6633574683409442
              }
            },
            "B (randomise)": {
              "archi1": {
                "mse_d2": 24.228759288787842,
                "mse_all": 13.350339561700821,
                "div_rate": 0.1,
                "delta_d2": 1.9908589001050352,
                "delta_all": 0.8522131811587041,
                "delta_transitoire": -0.4879683864402515,
                "delta_etabli": 3.2207764443034614,
                "delta_bloc_debut": 1.9927001335976962,
                "delta_bloc_fin": 1.9927001335976962
              },
              "archi2": {
                "mse_d2": 13.238725948333741,
                "mse_all": 8.368754583597184,
                "div_rate": 0.0,
                "delta_d2": -0.6339909322182866,
                "delta_all": -1.1761016164837061,
                "delta_transitoire": 0.15202756898285227,
                "delta_etabli": 0.7607553147907691,
                "delta_bloc_debut": -0.6347779311701678,
                "delta_bloc_fin": -0.6347779311701678
              }
            }
          }
        },
        "cadence_1": {
          "libelle": "ratio_gps = 1",
          "x": 1,
          "T": 160,
          "nominal": false,
          "reference": false,
          "ekf_d2": 6.5938722178339955,
          "ekf_all": 5.531605194509029,
          "modeles": {
            "A (etroit)": {
              "archi1": {
                "mse_d2": 4599949060338523.0,
                "mse_all": 1686457027386655.8,
                "div_rate": 0.15,
                "delta_d2": 148.43612495619803,
                "delta_all": 144.84124103626021,
                "delta_transitoire": -0.4541017219827733,
                "delta_etabli": 153.1896069445848,
                "delta_bloc_debut": 148.45423080336377,
                "delta_bloc_fin": 148.45423080336377
              },
              "archi2": {
                "mse_d2": 343.7313224568963,
                "mse_all": 1027.4584635868669,
                "div_rate": 0.15,
                "delta_d2": 17.17078582221722,
                "delta_all": 22.689130981727594,
                "delta_transitoire": -0.21715875369937546,
                "delta_etabli": 21.63485143678195,
                "delta_bloc_debut": 17.188545395761025,
                "delta_bloc_fin": 17.188545395761025
              }
            },
            "B (randomise)": {
              "archi1": {
                "mse_d2": 10528830444.358057,
                "mse_all": 4721991275.176492,
                "div_rate": 0.15,
                "delta_d2": 92.03239605163185,
                "delta_all": 89.31274004515494,
                "delta_transitoire": 0.2969708326614545,
                "delta_etabli": 96.78217999354884,
                "delta_bloc_debut": 92.05050218673938,
                "delta_bloc_fin": 92.05050218673938
              },
              "archi2": {
                "mse_d2": 120574.73990686536,
                "mse_all": 96113.3080622673,
                "div_rate": 0.15,
                "delta_d2": 42.62115807045619,
                "delta_all": 42.39932349429289,
                "delta_transitoire": -0.2414994485718488,
                "delta_etabli": 47.36283983578086,
                "delta_bloc_debut": 42.63926312871294,
                "delta_bloc_fin": 42.63926312871294
              }
            }
          }
        },
        "cadence_2": {
          "libelle": "ratio_gps = 2",
          "x": 2,
          "T": 160,
          "nominal": false,
          "reference": false,
          "ekf_d2": 9.862084567546844,
          "ekf_all": 8.898735719919205,
          "modeles": {
            "A (etroit)": {
              "archi1": {
                "mse_d2": 326766.25086595566,
                "mse_all": 271421.95441912336,
                "div_rate": 0.05,
                "delta_d2": 45.202684729975005,
                "delta_all": 44.843166643610424,
                "delta_transitoire": 1.6653071361709046,
                "delta_etabli": 49.4478073877965,
                "delta_bloc_debut": 45.2155188237169,
                "delta_bloc_fin": 45.2155188237169
              },
              "archi2": {
                "mse_d2": 5.886960327625275,
                "mse_all": 6.450057145953179,
                "div_rate": 0.0,
                "delta_d2": -2.2407761319330923,
                "delta_all": -1.3976474662276994,
                "delta_transitoire": 1.684664749471033,
                "delta_etabli": -2.3841070088917933,
                "delta_bloc_debut": -2.2494640223989757,
                "delta_bloc_fin": -2.2494640223989757
              }
            },
            "B (randomise)": {
              "archi1": {
                "mse_d2": 208.50997462421657,
                "mse_all": 93.47292186617851,
                "div_rate": 0.05,
                "delta_d2": 13.251581131341261,
                "delta_all": 10.213575094754404,
                "delta_transitoire": 1.1388270759100823,
                "delta_etabli": 16.57128622095321,
                "delta_bloc_debut": 13.263809467989827,
                "delta_bloc_fin": 13.263809467989827
              },
              "archi2": {
                "mse_d2": 8.368522584438324,
                "mse_all": 9.750742168724537,
                "div_rate": 0.0,
                "delta_d2": -0.7131992977977593,
                "delta_all": 0.39709363859931074,
                "delta_transitoire": 2.231804834423586,
                "delta_etabli": -1.3169852957804014,
                "delta_bloc_debut": -0.7154939285953961,
                "delta_bloc_fin": -0.7154939285953961
              }
            }
          }
        },
        "cadence_10": {
          "libelle": "ratio_gps = 10",
          "x": 10,
          "T": 160,
          "nominal": false,
          "reference": false,
          "ekf_d2": 12.697129353880882,
          "ekf_all": 8.660844069719314,
          "modeles": {
            "A (etroit)": {
              "archi1": {
                "mse_d2": 19.96616583764553,
                "mse_all": 10.280541369318962,
                "div_rate": 0.0,
                "delta_d2": 1.965891299619057,
                "delta_all": 0.7445576542966592,
                "delta_transitoire": 0.9541144613971065,
                "delta_etabli": 4.043686756513311,
                "delta_bloc_debut": 1.9683785340745152,
                "delta_bloc_fin": 1.9683785340745152
              },
              "archi2": {
                "mse_d2": 9.101469877362252,
                "mse_all": 6.416773909330368,
                "div_rate": 0.0,
                "delta_d2": -1.4459400798536133,
                "delta_all": -1.3024348212093073,
                "delta_transitoire": 0.7840013561756448,
                "delta_etabli": -0.5642229651663606,
                "delta_bloc_debut": -1.4486404354674731,
                "delta_bloc_fin": -1.4486404354674731
              }
            },
            "B (randomise)": {
              "archi1": {
                "mse_d2": 86.49910344481468,
                "mse_all": 37.91473466157913,
                "div_rate": 0.0,
                "delta_d2": 8.333060620148746,
                "delta_all": 6.412478014025483,
                "delta_transitoire": 0.9679170342114891,
                "delta_etabli": 10.296164901928691,
                "delta_bloc_debut": 8.338886977895424,
                "delta_bloc_fin": 8.338886977895424
              },
              "archi2": {
                "mse_d2": 7.124047935009003,
                "mse_all": 5.669265958666801,
                "div_rate": 0.0,
                "delta_d2": -2.5097871104781233,
                "delta_all": -1.840333883610682,
                "delta_transitoire": 0.7045243544353093,
                "delta_etabli": -2.4396269777634543,
                "delta_bloc_debut": -2.5151359280914267,
                "delta_bloc_fin": -2.5151359280914267
              }
            }
          }
        },
        "cadence_20": {
          "libelle": "ratio_gps = 20",
          "x": 20,
          "T": 160,
          "nominal": false,
          "reference": false,
          "ekf_d2": 16.189627468585968,
          "ekf_all": 10.441817182302476,
          "modeles": {
            "A (etroit)": {
              "archi1": {
                "mse_d2": 113.66716293096542,
                "mse_all": 44.209689831733705,
                "div_rate": 0.0,
                "delta_d2": 8.463981648463063,
                "delta_all": 6.2674138275921365,
                "delta_transitoire": -0.009073542164338068,
                "delta_etabli": 11.722117860229002,
                "delta_bloc_debut": 8.470532665793026,
                "delta_bloc_fin": 8.470532665793026
              },
              "archi2": {
                "mse_d2": 34.387830522656444,
                "mse_all": 22.38578594326973,
                "div_rate": 0.0,
                "delta_d2": 3.2716792215461714,
                "delta_all": 3.311962614306767,
                "delta_transitoire": -0.4158255788812765,
                "delta_etabli": 4.547647729557388,
                "delta_bloc_debut": 3.275722652717974,
                "delta_bloc_fin": 3.275722652717974
              }
            },
            "B (randomise)": {
              "archi1": {
                "mse_d2": 188.15733098983765,
                "mse_all": 106.22315809726715,
                "div_rate": 0.1,
                "delta_d2": 10.652842884407441,
                "delta_all": 10.074431239732256,
                "delta_transitoire": -0.07136444699802896,
                "delta_etabli": 10.994739534566733,
                "delta_bloc_debut": 10.659823772429938,
                "delta_bloc_fin": 10.659823772429938
              },
              "archi2": {
                "mse_d2": 26.061271286010744,
                "mse_all": 15.431337773799896,
                "div_rate": 0.0,
                "delta_d2": 2.0675874152929823,
                "delta_all": 1.6962749234469592,
                "delta_transitoire": 0.007857220536294314,
                "delta_etabli": 2.941798091436854,
                "delta_bloc_debut": 2.0704818554495015,
                "delta_bloc_fin": 2.0704818554495015
              }
            }
          }
        }
      }
    },
    "panne": {
      "titre": "Panne GPS + distances (seul l'accelerometre subsiste)",
      "xlabel": "Duree de la panne (s)",
      "groupe": "capteurs",
      "scenarios": {
        "panne_0s": {
          "libelle": "aucune panne",
          "x": 0.0,
          "T": 160,
          "nominal": true,
          "reference": true,
          "ekf_d2": 14.887526243925095,
          "ekf_all": 11.152833542227745,
          "modeles": {
            "A (etroit)": {
              "archi1": {
                "mse_d2": 35.115554162859915,
                "mse_all": 16.98645043373108,
                "div_rate": 0.0,
                "delta_d2": 3.7267698638930495,
                "delta_all": 1.8271741585487993,
                "delta_transitoire": 0.32925127254340153,
                "delta_etabli": 5.531687072810825,
                "delta_bloc_debut": 3.731584325239657,
                "delta_bloc_fin": 3.731584325239657
              },
              "archi2": {
                "mse_d2": 14.757737801969052,
                "mse_all": 9.159339198470116,
                "div_rate": 0.0,
                "delta_d2": -0.03802750024703936,
                "delta_all": -0.855210777611866,
                "delta_transitoire": 0.19241641502552492,
                "delta_etabli": 0.7997220319446321,
                "delta_bloc_debut": -0.038101109192631856,
                "delta_bloc_fin": -0.038101109192631856
              }
            },
            "B (randomise)": {
              "archi1": {
                "mse_d2": 35.02356380969286,
                "mse_all": 17.672965368628503,
                "div_rate": 0.0,
                "delta_d2": 3.7153779539772454,
                "delta_all": 1.9992420609082346,
                "delta_transitoire": 0.08374414882093699,
                "delta_etabli": 5.221757714741284,
                "delta_bloc_debut": 3.720183344342832,
                "delta_bloc_fin": 3.720183344342832
              },
              "archi2": {
                "mse_d2": 16.098922911286355,
                "mse_all": 9.464265209436416,
                "div_rate": 0.0,
                "delta_d2": 0.3397428072232633,
                "delta_all": -0.7129831866343201,
                "delta_transitoire": 0.9242207797755058,
                "delta_etabli": 1.1572051638524652,
                "delta_bloc_debut": 0.34037172635469914,
                "delta_bloc_fin": 0.34037172635469914
              }
            }
          }
        },
        "panne_2s": {
          "libelle": "panne de 2 s",
          "x": 2.0,
          "T": 160,
          "nominal": false,
          "reference": false,
          "ekf_d2": 10.599591398239136,
          "ekf_all": 7.928560829162597,
          "modeles": {
            "A (etroit)": {
              "archi1": {
                "mse_d2": 45.73543947339058,
                "mse_all": 20.635635554790497,
                "div_rate": 0.0,
                "delta_d2": 6.349637325447157,
                "delta_all": 4.154234866938616,
                "delta_transitoire": 1.169580730300953,
                "delta_etabli": 8.57532973044238,
                "delta_bloc_debut": 6.3577556055138595,
                "delta_bloc_fin": 6.3577556055138595
              },
              "archi2": {
                "mse_d2": 9.815553703904152,
                "mse_all": 8.537173475325108,
                "div_rate": 0.0,
                "delta_d2": -0.3337432049107933,
                "delta_all": 0.3211974385709823,
                "delta_transitoire": 1.1490995162640474,
                "delta_etabli": 0.760806362837626,
                "delta_bloc_debut": -0.3345879922651219,
                "delta_bloc_fin": -0.3345879922651219
              }
            },
            "B (randomise)": {
              "archi1": {
                "mse_d2": 25.152660773694514,
                "mse_all": 14.55354298055172,
                "div_rate": 0.0,
                "delta_d2": 3.752948096079746,
                "delta_all": 2.637743702968592,
                "delta_transitoire": 0.6596332286941544,
                "delta_etabli": 4.528012162752514,
                "delta_bloc_debut": 3.7590635449725687,
                "delta_bloc_fin": 3.7590635449725687
              },
              "archi2": {
                "mse_d2": 11.737158143520356,
                "mse_all": 9.3036072909832,
                "div_rate": 0.0,
                "delta_d2": 0.4427383215005068,
                "delta_all": 0.6945700784592334,
                "delta_transitoire": 2.0329753464781084,
                "delta_etabli": 1.4193961439214087,
                "delta_bloc_debut": 0.44376350482297827,
                "delta_bloc_fin": 0.44376350482297827
              }
            }
          }
        },
        "panne_5s": {
          "libelle": "panne de 5 s",
          "x": 5.0,
          "T": 160,
          "nominal": false,
          "reference": false,
          "ekf_d2": 10.47689909040928,
          "ekf_all": 6.794739472866058,
          "modeles": {
            "A (etroit)": {
              "archi1": {
                "mse_d2": 49.456384509801865,
                "mse_all": 21.25054116845131,
                "div_rate": 0.0,
                "delta_d2": 6.739896032844119,
                "delta_all": 4.951971848527082,
                "delta_transitoire": 0.2928540486104402,
                "delta_etabli": 7.856714254009724,
                "delta_bloc_debut": 6.747462669177374,
                "delta_bloc_fin": 6.747462669177374
              },
              "archi2": {
                "mse_d2": 9.820272967219353,
                "mse_all": 6.166031962633133,
                "div_rate": 0.0,
                "delta_d2": -0.28109201235295594,
                "delta_all": -0.4216703782941259,
                "delta_transitoire": 0.7473550707164299,
                "delta_etabli": 0.23552243395057448,
                "delta_bloc_debut": -0.2817341587995501,
                "delta_bloc_fin": -0.2817341587995501
              }
            },
            "B (randomise)": {
              "archi1": {
                "mse_d2": 33.06717823147774,
                "mse_all": 16.77461705207825,
                "div_rate": 0.0,
                "delta_d2": 4.991643752583117,
                "delta_all": 3.92479804978503,
                "delta_transitoire": 0.37016384301020994,
                "delta_etabli": 5.501674934114556,
                "delta_bloc_debut": 4.998202976265164,
                "delta_bloc_fin": 4.998202976265164
              },
              "archi2": {
                "mse_d2": 14.603400933742524,
                "mse_all": 8.34791710972786,
                "div_rate": 0.0,
                "delta_d2": 1.4422124788175887,
                "delta_all": 0.8940531864498645,
                "delta_transitoire": 2.3275608840967554,
                "delta_etabli": 2.026089742046622,
                "delta_bloc_debut": 1.4449267902208585,
                "delta_bloc_fin": 1.4449267902208585
              }
            }
          }
        },
        "panne_10s": {
          "libelle": "panne de 10 s",
          "x": 10.0,
          "T": 160,
          "nominal": false,
          "reference": false,
          "ekf_d2": 210.51251468658447,
          "ekf_all": 77.46906266212463,
          "modeles": {
            "A (etroit)": {
              "archi1": {
                "mse_d2": 108.16883864402772,
                "mse_all": 42.98174911737442,
                "div_rate": 0.0,
                "delta_d2": -2.8917575223846064,
                "delta_all": -2.5584421636976895,
                "delta_transitoire": 0.48967521172115347,
                "delta_etabli": 2.7149772381734376,
                "delta_bloc_debut": -2.892114368232682,
                "delta_bloc_fin": -2.892114368232682
              },
              "archi2": {
                "mse_d2": 30.424894630908966,
                "mse_all": 17.638539052009584,
                "div_rate": 0.0,
                "delta_d2": -8.400488363301907,
                "delta_all": -6.426656902883673,
                "delta_transitoire": -0.1151831701637904,
                "delta_etabli": -3.913098063370879,
                "delta_bloc_debut": -8.402721305473884,
                "delta_bloc_fin": -8.402721305473884
              }
            },
            "B (randomise)": {
              "archi1": {
                "mse_d2": 607.6800923228263,
                "mse_all": 252.42427365779878,
                "div_rate": 0.0,
                "delta_d2": 4.603970898743856,
                "delta_all": 5.130028140189593,
                "delta_transitoire": 0.8879400713989029,
                "delta_etabli": 10.610802599188432,
                "delta_bloc_debut": 4.604217499517226,
                "delta_bloc_fin": 4.604217499517226
              },
              "archi2": {
                "mse_d2": 57.05320582389832,
                "mse_all": 28.740482354164122,
                "div_rate": 0.0,
                "delta_d2": -5.669978666842146,
                "delta_all": -4.306342485834077,
                "delta_transitoire": 0.47205953333061773,
                "delta_etabli": -0.7985866608825152,
                "delta_bloc_debut": -5.670993168329299,
                "delta_bloc_fin": -5.670993168329299
              }
            }
          }
        }
      }
    },
    "aberrations": {
      "titre": "Mesures aberrantes (bruit x10 sur une fraction des mesures)",
      "xlabel": "Taux de mesures aberrantes",
      "groupe": "capteurs",
      "scenarios": {
        "aberrations_0": {
          "libelle": "aucune aberration",
          "x": 0.0,
          "T": 160,
          "nominal": true,
          "reference": true,
          "ekf_d2": 10.456278206408024,
          "ekf_all": 6.917049540579319,
          "modeles": {
            "A (etroit)": {
              "archi1": {
                "mse_d2": 13.735193157196045,
                "mse_all": 8.123614270985126,
                "div_rate": 0.0,
                "delta_d2": 1.1845764141449435,
                "delta_all": 0.6982840771231202,
                "delta_transitoire": 0.406855840540417,
                "delta_etabli": 2.5427692463848923,
                "delta_bloc_debut": 1.1876585627115177,
                "delta_bloc_fin": 1.1876585627115177
              },
              "archi2": {
                "mse_d2": 8.787550783157348,
                "mse_all": 6.00340821146965,
                "div_rate": 0.0,
                "delta_d2": -0.7550928202416519,
                "delta_all": -0.6152301123203346,
                "delta_transitoire": 0.6553079903010178,
                "delta_etabli": -0.4885402589778235,
                "delta_bloc_debut": -0.7575460927350158,
                "delta_bloc_fin": -0.7575460927350158
              }
            },
            "B (randomise)": {
              "archi1": {
                "mse_d2": 18.418412567675112,
                "mse_all": 11.539810852706433,
                "div_rate": 0.0,
                "delta_d2": 2.45875066984527,
                "delta_all": 2.2227780427391433,
                "delta_transitoire": 0.3063855385720652,
                "delta_etabli": 2.4568252958779797,
                "delta_bloc_debut": 2.4643303883999317,
                "delta_bloc_fin": 2.4643303883999317
              },
              "archi2": {
                "mse_d2": 7.239042952656746,
                "mse_all": 5.284653744101524,
                "div_rate": 0.0,
                "delta_d2": -1.5969597633210808,
                "delta_all": -1.1690434899038806,
                "delta_transitoire": 1.3858295841946777,
                "delta_etabli": -0.7260279220645106,
                "delta_bloc_debut": -1.6027036684396478,
                "delta_bloc_fin": -1.6027036684396478
              }
            }
          }
        },
        "aberrations_0.01": {
          "libelle": "1 % aberrantes",
          "x": 0.01,
          "T": 160,
          "nominal": false,
          "reference": false,
          "ekf_d2": 15.690847969055175,
          "ekf_all": 14.245314824581147,
          "modeles": {
            "A (etroit)": {
              "archi1": {
                "mse_d2": 24.921875089406967,
                "mse_all": 17.940993320941924,
                "div_rate": 0.0,
                "delta_d2": 2.00934300486185,
                "delta_all": 1.001744328349097,
                "delta_transitoire": -0.6770951570520167,
                "delta_etabli": 3.6648200120312917,
                "delta_bloc_debut": 2.011210616344149,
                "delta_bloc_fin": 2.011210616344149
              },
              "archi2": {
                "mse_d2": 7.435828176140785,
                "mse_all": 5.885262659192085,
                "div_rate": 0.0,
                "delta_d2": -3.243170687519336,
                "delta_all": -3.8390620151604993,
                "delta_transitoire": -1.184235963453369,
                "delta_etabli": -2.7088662232000122,
                "delta_bloc_debut": -3.248773372895957,
                "delta_bloc_fin": -3.248773372895957
              }
            },
            "B (randomise)": {
              "archi1": {
                "mse_d2": 16.27320048213005,
                "mse_all": 13.101684957742691,
                "div_rate": 0.0,
                "delta_d2": 0.1582656040022089,
                "delta_all": -0.363448994799892,
                "delta_transitoire": -0.3790880725718598,
                "delta_etabli": 1.570863774357285,
                "delta_bloc_debut": 0.15844587211755046,
                "delta_bloc_fin": 0.15844587211755046
              },
              "archi2": {
                "mse_d2": 7.20332200229168,
                "mse_all": 8.10322334319353,
                "div_rate": 0.0,
                "delta_d2": -3.3811358547529,
                "delta_all": -2.450142424729205,
                "delta_transitoire": -0.08606552864897052,
                "delta_etabli": -3.4543141086571234,
                "delta_bloc_debut": -3.3870826971263868,
                "delta_bloc_fin": -3.3870826971263868
              }
            }
          }
        },
        "aberrations_0.02": {
          "libelle": "2 % aberrantes",
          "x": 0.02,
          "T": 160,
          "nominal": false,
          "reference": false,
          "ekf_d2": 20.269722467660905,
          "ekf_all": 12.156867164373399,
          "modeles": {
            "A (etroit)": {
              "archi1": {
                "mse_d2": 18.953272598981858,
                "mse_all": 9.586238569021225,
                "div_rate": 0.0,
                "delta_d2": -0.29163593416043665,
                "delta_all": -1.031734383659743,
                "delta_transitoire": 0.955020324894802,
                "delta_etabli": 1.3464857111115915,
                "delta_bloc_debut": -0.29208973803399924,
                "delta_bloc_fin": -0.29208973803399924
              },
              "archi2": {
                "mse_d2": 10.285056360065937,
                "mse_all": 6.862329778075218,
                "div_rate": 0.0,
                "delta_d2": -2.9464112650392824,
                "delta_all": -2.4835008647444416,
                "delta_transitoire": 0.6041389519208387,
                "delta_etabli": -2.3184717044105927,
                "delta_bloc_debut": -2.9527542939987694,
                "delta_bloc_fin": -2.9527542939987694
              }
            },
            "B (randomise)": {
              "archi1": {
                "mse_d2": 25.88938205242157,
                "mse_all": 11.929958283901215,
                "div_rate": 0.0,
                "delta_d2": 1.0627388209692734,
                "delta_all": -0.08182746222326659,
                "delta_transitoire": 0.5830192088481397,
                "delta_etabli": 2.4093967512207444,
                "delta_bloc_debut": 1.0641557381193936,
                "delta_bloc_fin": 1.0641557381193936
              },
              "archi2": {
                "mse_d2": 9.089187279343605,
                "mse_all": 5.531363901495934,
                "div_rate": 0.0,
                "delta_d2": -3.4832275039621603,
                "delta_all": -3.4198944017696276,
                "delta_transitoire": 0.7284797677995116,
                "delta_etabli": -2.642075901672014,
                "delta_bloc_debut": -3.491266406716063,
                "delta_bloc_fin": -3.491266406716063
              }
            }
          }
        },
        "aberrations_0.05": {
          "libelle": "5 % aberrantes",
          "x": 0.05,
          "T": 160,
          "nominal": false,
          "reference": false,
          "ekf_d2": 49.60257901549339,
          "ekf_all": 32.48351573348045,
          "modeles": {
            "A (etroit)": {
              "archi1": {
                "mse_d2": 58.03413850069046,
                "mse_all": 32.81825112700462,
                "div_rate": 0.0,
                "delta_d2": 0.6817928425331135,
                "delta_all": 0.044524065604185536,
                "delta_transitoire": -0.4187209004438572,
                "delta_etabli": 1.5586518158828697,
                "delta_bloc_debut": 0.6821272216600964,
                "delta_bloc_fin": 0.6821272216600964
              },
              "archi2": {
                "mse_d2": 10.081405359506608,
                "mse_all": 10.736859494447708,
                "div_rate": 0.0,
                "delta_d2": -6.919831801205456,
                "delta_all": -4.807857575263926,
                "delta_transitoire": -0.2080341283542479,
                "delta_etabli": -6.736243983803116,
                "delta_bloc_debut": -6.9288654052151,
                "delta_bloc_fin": -6.9288654052151
              }
            },
            "B (randomise)": {
              "archi1": {
                "mse_d2": 59.84498608112335,
                "mse_all": 44.43610640764236,
                "div_rate": 0.0,
                "delta_d2": 0.8152351270539854,
                "delta_all": 1.3607297069513848,
                "delta_transitoire": -0.24402130614300713,
                "delta_etabli": 2.3933088577672668,
                "delta_bloc_debut": 0.8156291825934003,
                "delta_bloc_fin": 0.8156291825934003
              },
              "archi2": {
                "mse_d2": 15.498501388728618,
                "mse_all": 19.677981109917162,
                "div_rate": 0.0,
                "delta_d2": -5.052145510478421,
                "delta_all": -2.176824880448283,
                "delta_transitoire": -0.6494872669587413,
                "delta_etabli": -4.29934303502568,
                "delta_bloc_debut": -5.057214015057209,
                "delta_bloc_fin": -5.057214015057209
              }
            }
          }
        }
      }
    },
    "bruit_r": {
      "titre": "Bruit de mesure, y compris hors de la plage d'entrainement",
      "xlabel": "$1/r^2$ [dB]",
      "groupe": "bruit",
      "scenarios": {
        "bruit_r_-30dB": {
          "libelle": "-30 dB",
          "x": -30,
          "T": 160,
          "nominal": false,
          "reference": false,
          "ekf_d2": 536.6041297912598,
          "ekf_all": 483.38536071777344,
          "modeles": {
            "A (etroit)": {
              "archi1": {
                "mse_d2": 6417207.455395508,
                "mse_all": 6369995.231347656,
                "div_rate": 0.85,
                "delta_d2": 40.776920686356334,
                "delta_all": 41.198456135161656,
                "delta_transitoire": -2.6932288357363143,
                "delta_etabli": 49.27498679753532,
                "delta_bloc_debut": 40.77709102108672,
                "delta_bloc_fin": 40.77709102108672
              },
              "archi2": {
                "mse_d2": 2263805253857.9688,
                "mse_all": 3614268973027.3374,
                "div_rate": 1.0,
                "delta_d2": 96.25185052751418,
                "delta_all": 98.73726975817029,
                "delta_transitoire": 1.1212405138560586,
                "delta_etabli": 105.03250719304368,
                "delta_bloc_debut": 96.25202081583467,
                "delta_bloc_fin": 96.25202081583467
              }
            },
            "B (randomise)": {
              "archi1": {
                "mse_d2": 112099.65766601563,
                "mse_all": 64902.975,
                "div_rate": 0.65,
                "delta_d2": 23.199502755842015,
                "delta_all": 21.279711105946596,
                "delta_transitoire": -0.26658274564088363,
                "delta_etabli": 30.152707862882707,
                "delta_bloc_debut": 23.19967217311391,
                "delta_bloc_fin": 23.19967217311391
              },
              "archi2": {
                "mse_d2": 1827290248.2785156,
                "mse_all": 1830669185.6148438,
                "div_rate": 1.0,
                "delta_d2": 65.3215352577415,
                "delta_all": 65.7831637772077,
                "delta_transitoire": -3.364444443099695,
                "delta_etabli": 73.68191188659343,
                "delta_bloc_debut": 65.32170537714843,
                "delta_bloc_fin": 65.32170537714843
              }
            }
          }
        },
        "bruit_r_-20dB": {
          "libelle": "-20 dB",
          "x": -20,
          "T": 160,
          "nominal": false,
          "reference": false,
          "ekf_d2": 264.10595550537107,
          "ekf_all": 167.30189914703368,
          "modeles": {
            "A (etroit)": {
              "archi1": {
                "mse_d2": 2635.171727657318,
                "mse_all": 1368.0909184455872,
                "div_rate": 0.05,
                "delta_d2": 9.990307278536347,
                "delta_all": 9.1261408907142,
                "delta_transitoire": -3.5726225471539177,
                "delta_etabli": 14.888099732035984,
                "delta_bloc_debut": 9.990564506043752,
                "delta_bloc_fin": 9.990564506043752
              },
              "archi2": {
                "mse_d2": 35173229.28843956,
                "mse_all": 57425760.74887705,
                "div_rate": 0.25,
                "delta_d2": 51.244340486269266,
                "delta_all": 55.356058862948544,
                "delta_transitoire": -4.221040297150118,
                "delta_etabli": 57.5403605908751,
                "delta_bloc_debut": 51.244626303812346,
                "delta_bloc_fin": 51.244626303812346
              }
            },
            "B (randomise)": {
              "archi1": {
                "mse_d2": 1947.3038621902465,
                "mse_all": 1016.3056383132935,
                "div_rate": 0.05,
                "delta_d2": 8.67655530690843,
                "delta_all": 7.835234640520118,
                "delta_transitoire": -2.6218014798375218,
                "delta_etabli": 12.919031234798526,
                "delta_bloc_debut": 8.676802535478886,
                "delta_bloc_fin": 8.676802535478886
              },
              "archi2": {
                "mse_d2": 17115233.026207376,
                "mse_all": 14709712.262167264,
                "div_rate": 0.05,
                "delta_d2": 48.116046219722264,
                "delta_all": 49.441033066073956,
                "delta_transitoire": -6.359713863271471,
                "delta_etabli": 54.298823623908035,
                "delta_bloc_debut": 48.11633183458201,
                "delta_bloc_fin": 48.11633183458201
              }
            }
          }
        },
        "bruit_r_-10dB": {
          "libelle": "-10 dB",
          "x": -10,
          "T": 160,
          "nominal": false,
          "reference": false,
          "ekf_d2": 54.578452014923094,
          "ekf_all": 42.10248218774795,
          "modeles": {
            "A (etroit)": {
              "archi1": {
                "mse_d2": 23.55618108510971,
                "mse_all": 13.009968018531799,
                "div_rate": 0.0,
                "delta_d2": -3.6491632955823223,
                "delta_all": -5.100314718240364,
                "delta_transitoire": -0.6560483633335399,
                "delta_etabli": -2.7230560086209183,
                "delta_bloc_debut": -3.6529799971048575,
                "delta_bloc_fin": -3.6529799971048575
              },
              "archi2": {
                "mse_d2": 26.749807369709014,
                "mse_all": 16.9135511636734,
                "div_rate": 0.0,
                "delta_d2": -3.097005548676852,
                "delta_all": -3.96072899293011,
                "delta_transitoire": -0.6621470973066225,
                "delta_etabli": -1.955184126308302,
                "delta_bloc_debut": -3.1000203167903435,
                "delta_bloc_fin": -3.1000203167903435
              }
            },
            "B (randomise)": {
              "archi1": {
                "mse_d2": 31.700867116451263,
                "mse_all": 15.158687734603882,
                "div_rate": 0.0,
                "delta_d2": -2.359500721283614,
                "delta_all": -4.436460941060354,
                "delta_transitoire": -0.3966391524566882,
                "delta_etabli": -1.209456682989169,
                "delta_bloc_debut": -2.3615916526640017,
                "delta_bloc_fin": -2.3615916526640017
              },
              "archi2": {
                "mse_d2": 13.538229608535767,
                "mse_all": 10.44707931280136,
                "div_rate": 0.0,
                "delta_d2": -6.054593383383205,
                "delta_all": -6.053128089694572,
                "delta_transitoire": -0.4611441448317273,
                "delta_etabli": -5.141755130966864,
                "delta_bloc_debut": -6.063383738227895,
                "delta_bloc_fin": -6.063383738227895
              }
            }
          }
        },
        "bruit_r_+0dB": {
          "libelle": "+0 dB",
          "x": 0,
          "T": 160,
          "nominal": true,
          "reference": true,
          "ekf_d2": 12.718521022796631,
          "ekf_all": 9.297827565670014,
          "modeles": {
            "A (etroit)": {
              "archi1": {
                "mse_d2": 302.35486063957217,
                "mse_all": 159.69679445028305,
                "div_rate": 0.05,
                "delta_d2": 13.760803425129644,
                "delta_all": 12.3491471110777,
                "delta_transitoire": 0.9923801470676646,
                "delta_etabli": 16.404977613205375,
                "delta_bloc_debut": 13.772660223686987,
                "delta_bloc_fin": 13.772660223686987
              },
              "archi2": {
                "mse_d2": 15.396629311144352,
                "mse_all": 9.591865257918835,
                "div_rate": 0.0,
                "delta_d2": 0.8298904165812955,
                "delta_all": 0.1352158172105209,
                "delta_transitoire": 0.8778236404995737,
                "delta_etabli": 0.8689577027342079,
                "delta_bloc_debut": 0.8320458417345522,
                "delta_bloc_fin": 0.8320458417345522
              }
            },
            "B (randomise)": {
              "archi1": {
                "mse_d2": 55.01820082217455,
                "mse_all": 27.47426161468029,
                "div_rate": 0.05,
                "delta_d2": 6.360697720252908,
                "delta_all": 4.705445417135191,
                "delta_transitoire": 0.6674594516924098,
                "delta_etabli": 6.914842502522715,
                "delta_bloc_debut": 6.3702163156316765,
                "delta_bloc_fin": 6.3702163156316765
              },
              "archi2": {
                "mse_d2": 18.388256630301477,
                "mse_all": 11.561042401194573,
                "div_rate": 0.0,
                "delta_d2": 1.6010394404536763,
                "delta_all": 0.9461550635982481,
                "delta_transitoire": 1.3335157012939067,
                "delta_etabli": 2.0937976880207576,
                "delta_bloc_debut": 1.6048595460054196,
                "delta_bloc_fin": 1.6048595460054196
              }
            }
          }
        },
        "bruit_r_+10dB": {
          "libelle": "+10 dB",
          "x": 10,
          "T": 160,
          "nominal": false,
          "reference": false,
          "ekf_d2": 6.009766671806574,
          "ekf_all": 5.138481318205595,
          "modeles": {
            "A (etroit)": {
              "archi1": {
                "mse_d2": 37.586312206089495,
                "mse_all": 18.135004974901676,
                "div_rate": 0.05,
                "delta_d2": 7.9617210593317385,
                "delta_all": 5.476828971371387,
                "delta_transitoire": 1.1311515047174978,
                "delta_etabli": 9.625753803185734,
                "delta_bloc_debut": 7.976939579424628,
                "delta_bloc_fin": 7.976939579424628
              },
              "archi2": {
                "mse_d2": 12.893414187431336,
                "mse_all": 12.280975198745727,
                "div_rate": 0.0,
                "delta_d2": 3.3151032322473184,
                "delta_all": 3.7839807239721313,
                "delta_transitoire": 1.8786303244733724,
                "delta_etabli": 3.7322511528482885,
                "delta_bloc_debut": 3.3247808022094665,
                "delta_bloc_fin": 3.3247808022094665
              }
            },
            "B (randomise)": {
              "archi1": {
                "mse_d2": 10.398719404637813,
                "mse_all": 8.130239926278591,
                "div_rate": 0.0,
                "delta_d2": 2.3812224859691273,
                "delta_all": 1.992685800506024,
                "delta_transitoire": 1.2031576649269846,
                "delta_etabli": 1.0128116989372462,
                "delta_bloc_debut": 2.3888747701259394,
                "delta_bloc_fin": 2.3888747701259394
              },
              "archi2": {
                "mse_d2": 7.014395385980606,
                "mse_all": 6.131517733633518,
                "div_rate": 0.0,
                "delta_d2": 0.6713263153762659,
                "delta_all": 0.7673320675362676,
                "delta_transitoire": 2.609867965521437,
                "delta_etabli": 0.8913858738385083,
                "delta_bloc_debut": 0.6739243975892839,
                "delta_bloc_fin": 0.6739243975892839
              }
            }
          }
        },
        "bruit_r_+20dB": {
          "libelle": "+20 dB",
          "x": 20,
          "T": 160,
          "nominal": false,
          "reference": false,
          "ekf_d2": 6.303540784865618,
          "ekf_all": 4.796529131382703,
          "modeles": {
            "A (etroit)": {
              "archi1": {
                "mse_d2": 15.750518476963043,
                "mse_all": 8.307322254776954,
                "div_rate": 0.0,
                "delta_d2": 3.9771028737505434,
                "delta_all": 2.3853397132258953,
                "delta_transitoire": 0.7258327022422897,
                "delta_etabli": 6.485125201487584,
                "delta_bloc_debut": 3.994506091163302,
                "delta_bloc_fin": 3.994506091163302
              },
              "archi2": {
                "mse_d2": 29.99493536949158,
                "mse_all": 22.630930256843566,
                "div_rate": 0.05,
                "delta_d2": 6.774633633639499,
                "delta_all": 6.737753197643187,
                "delta_transitoire": 2.1007788804571303,
                "delta_etabli": 8.108025326755218,
                "delta_bloc_debut": 6.797537075686543,
                "delta_bloc_fin": 6.797537075686543
              }
            },
            "B (randomise)": {
              "archi1": {
                "mse_d2": 27.814242739230394,
                "mse_all": 14.109835423529148,
                "div_rate": 0.05,
                "delta_d2": 6.446826733591916,
                "delta_all": 4.685948617220182,
                "delta_transitoire": 0.3918094944560144,
                "delta_etabli": 7.98276868569539,
                "delta_bloc_debut": 6.469253982138171,
                "delta_bloc_fin": 6.469253982138171
              },
              "archi2": {
                "mse_d2": 19.30702340900898,
                "mse_all": 10.996959692239761,
                "div_rate": 0.0,
                "delta_d2": 4.861307560181486,
                "delta_all": 3.6034554674270733,
                "delta_transitoire": 1.7490650245983272,
                "delta_etabli": 6.136111171257426,
                "delta_bloc_debut": 4.880845362736077,
                "delta_bloc_fin": 4.880845362736077
              }
            }
          }
        },
        "bruit_r_+30dB": {
          "libelle": "+30 dB",
          "x": 30,
          "T": 160,
          "nominal": false,
          "reference": false,
          "ekf_d2": 3.918712792918086,
          "ekf_all": 3.885401065647602,
          "modeles": {
            "A (etroit)": {
              "archi1": {
                "mse_d2": 15.478242536261678,
                "mse_all": 9.867075945436955,
                "div_rate": 0.1,
                "delta_d2": 5.965782128629726,
                "delta_all": 4.047526159625921,
                "delta_transitoire": 0.43815422830770256,
                "delta_etabli": 9.327741059094073,
                "delta_bloc_debut": 5.984972849360082,
                "delta_bloc_fin": 5.984972849360082
              },
              "archi2": {
                "mse_d2": 16.43730574250221,
                "mse_all": 17.849286979436876,
                "div_rate": 0.0,
                "delta_d2": 6.2268719864476205,
                "delta_all": 6.621850171988159,
                "delta_transitoire": 1.8411032934671705,
                "delta_etabli": 8.521080883829647,
                "delta_bloc_debut": 6.2464409880995735,
                "delta_bloc_fin": 6.2464409880995735
              }
            },
            "B (randomise)": {
              "archi1": {
                "mse_d2": 9.896238005906344,
                "mse_all": 8.631682428717614,
                "div_rate": 0.0,
                "delta_d2": 4.023266968998853,
                "delta_all": 3.466595987332828,
                "delta_transitoire": 0.18851589294409052,
                "delta_etabli": 3.9242329984237347,
                "delta_bloc_debut": 4.038794406822995,
                "delta_bloc_fin": 4.038794406822995
              },
              "archi2": {
                "mse_d2": 6.602126818150282,
                "mse_all": 7.183809278160334,
                "div_rate": 0.0,
                "delta_d2": 2.2654042760899746,
                "delta_all": 2.6691893884183178,
                "delta_transitoire": 0.9794668808666479,
                "delta_etabli": 2.771514259523078,
                "delta_bloc_debut": 2.2758590244513948,
                "delta_bloc_fin": 2.2758590244513948
              }
            }
          }
        },
        "bruit_r_+40dB": {
          "libelle": "+40 dB",
          "x": 40,
          "T": 160,
          "nominal": false,
          "reference": false,
          "ekf_d2": 3.8700261883437634,
          "ekf_all": 2.9684924826025965,
          "modeles": {
            "A (etroit)": {
              "archi1": {
                "mse_d2": 26.16595750898123,
                "mse_all": 13.081093637645244,
                "div_rate": 0.05,
                "delta_d2": 8.300227279190755,
                "delta_all": 6.441081009996137,
                "delta_transitoire": 0.8193738145599394,
                "delta_etabli": 10.174561654611544,
                "delta_bloc_debut": 8.321740441231572,
                "delta_bloc_fin": 8.321740441231572
              },
              "archi2": {
                "mse_d2": 44.75144856572151,
                "mse_all": 31.982882368564606,
                "div_rate": 0.15,
                "delta_d2": 10.630931937308105,
                "delta_all": 10.32381647301582,
                "delta_transitoire": 2.4062201949168385,
                "delta_etabli": 12.986647801873598,
                "delta_bloc_debut": 10.653991904751221,
                "delta_bloc_fin": 10.653991904751221
              }
            },
            "B (randomise)": {
              "archi1": {
                "mse_d2": 13.591923454403878,
                "mse_all": 7.659217561781406,
                "div_rate": 0.0,
                "delta_d2": 5.455670161671348,
                "delta_all": 4.11648452472627,
                "delta_transitoire": 0.16143255124731995,
                "delta_etabli": 8.133125043705139,
                "delta_bloc_debut": 5.4737362336757025,
                "delta_bloc_fin": 5.4737362336757025
              },
              "archi2": {
                "mse_d2": 10.104243839532137,
                "mse_all": 5.439072397351265,
                "div_rate": 0.0,
                "delta_d2": 4.167899143508925,
                "delta_all": 2.629888861184864,
                "delta_transitoire": 1.2851497175767315,
                "delta_etabli": 6.5210845764271586,
                "delta_bloc_debut": 4.183487397248777,
                "delta_bloc_fin": 4.183487397248777
              }
            }
          }
        },
        "bruit_r_+50dB": {
          "libelle": "+50 dB",
          "x": 50,
          "T": 160,
          "nominal": false,
          "reference": false,
          "ekf_d2": 4.999121821671724,
          "ekf_all": 4.000580275058747,
          "modeles": {
            "A (etroit)": {
              "archi1": {
                "mse_d2": 19.929064962267876,
                "mse_all": 11.698836717009545,
                "div_rate": 0.0,
                "delta_d2": 6.005932027859516,
                "delta_all": 4.660196901585958,
                "delta_transitoire": 0.5587620625033684,
                "delta_etabli": 8.080173670572522,
                "delta_bloc_debut": 6.023985046310393,
                "delta_bloc_fin": 6.023985046310393
              },
              "archi2": {
                "mse_d2": 34.39136084318161,
                "mse_all": 27.923964935541154,
                "div_rate": 0.0,
                "delta_d2": 8.375556408683574,
                "delta_all": 8.438540946633372,
                "delta_transitoire": 2.962272002807589,
                "delta_etabli": 11.110040273897319,
                "delta_bloc_debut": 8.396145491639981,
                "delta_bloc_fin": 8.396145491639981
              }
            },
            "B (randomise)": {
              "archi1": {
                "mse_d2": 14.431007459759712,
                "mse_all": 9.001004469394683,
                "div_rate": 0.0,
                "delta_d2": 4.604029311501735,
                "delta_all": 3.521679880249437,
                "delta_transitoire": 0.7005096311322492,
                "delta_etabli": 5.936457489233008,
                "delta_bloc_debut": 4.619783732167695,
                "delta_bloc_fin": 4.619783732167695
              },
              "archi2": {
                "mse_d2": 12.2195312961936,
                "mse_all": 7.036081241071225,
                "div_rate": 0.0,
                "delta_d2": 3.8816082798313465,
                "delta_all": 2.4520785600881285,
                "delta_transitoire": 2.0394034497455507,
                "delta_etabli": 5.582662144274494,
                "delta_bloc_debut": 3.8958539152959126,
                "delta_bloc_fin": 3.8958539152959126
              }
            }
          }
        }
      }
    },
    "bruit_q": {
      "titre": "Bruit de process (jamais varie a l'entrainement)",
      "xlabel": "Facteur sur l'ecart-type du bruit de process",
      "groupe": "bruit",
      "scenarios": {
        "bruit_q_0.2": {
          "libelle": "q x 0.2",
          "x": 0.2,
          "T": 160,
          "nominal": false,
          "reference": false,
          "ekf_d2": 10.357546493411064,
          "ekf_all": 7.285761806368828,
          "modeles": {
            "A (etroit)": {
              "archi1": {
                "mse_d2": 33.89744201898575,
                "mse_all": 15.706283466517926,
                "div_rate": 0.0,
                "delta_d2": 5.149100350975955,
                "delta_all": 3.3359846258505508,
                "delta_transitoire": 1.4291887302256614,
                "delta_etabli": 6.797204448438333,
                "delta_bloc_debut": 5.157395379584802,
                "delta_bloc_fin": 5.157395379584802
              },
              "archi2": {
                "mse_d2": 8.712880370020866,
                "mse_all": 5.735655283927917,
                "div_rate": 0.0,
                "delta_d2": -0.750951403493906,
                "delta_all": -1.0389192663115239,
                "delta_transitoire": 1.1269522192500678,
                "delta_etabli": -0.04711106891695758,
                "delta_bloc_debut": -0.7532087231529961,
                "delta_bloc_fin": -0.7532087231529961
              }
            },
            "B (randomise)": {
              "archi1": {
                "mse_d2": 25.320269060134887,
                "mse_all": 12.964093144237996,
                "div_rate": 0.0,
                "delta_d2": 3.882114248473967,
                "delta_all": 2.5026717394904847,
                "delta_transitoire": 0.8509316746198959,
                "delta_etabli": 5.308467137080087,
                "delta_bloc_debut": 3.889173957237549,
                "delta_bloc_fin": 3.889173957237549
              },
              "archi2": {
                "mse_d2": 6.4199081540107725,
                "mse_all": 4.781822508573532,
                "div_rate": 0.0,
                "delta_d2": -2.077280765467501,
                "delta_all": -1.8288151677526623,
                "delta_transitoire": 1.8214217177740184,
                "delta_etabli": -1.8914796553582147,
                "delta_bloc_debut": -2.0846202311150757,
                "delta_bloc_fin": -2.0846202311150757
              }
            }
          }
        },
        "bruit_q_1": {
          "libelle": "q x 1",
          "x": 1.0,
          "T": 160,
          "nominal": true,
          "reference": true,
          "ekf_d2": 12.428774872422219,
          "ekf_all": 7.562000423669815,
          "modeles": {
            "A (etroit)": {
              "archi1": {
                "mse_d2": 28.43840582370758,
                "mse_all": 15.19827995300293,
                "div_rate": 0.0,
                "delta_d2": 3.5947692589209757,
                "delta_all": 3.031577426134671,
                "delta_transitoire": 1.254029981707665,
                "delta_etabli": 5.350509700378863,
                "delta_bloc_debut": 3.599548861427639,
                "delta_bloc_fin": 3.599548861427639
              },
              "archi2": {
                "mse_d2": 7.727710545063019,
                "mse_all": 5.896540945768356,
                "div_rate": 0.0,
                "delta_d2": -2.0637747508088906,
                "delta_all": -1.0803937871739555,
                "delta_transitoire": 0.6668116004149156,
                "delta_etabli": -1.3415400519108467,
                "delta_bloc_debut": -2.068945499229387,
                "delta_bloc_fin": -2.068945499229387
              }
            },
            "B (randomise)": {
              "archi1": {
                "mse_d2": 36.65682321190834,
                "mse_all": 18.9815595343709,
                "div_rate": 0.0,
                "delta_d2": 4.69726503487805,
                "delta_all": 3.99695194049414,
                "delta_transitoire": 0.3839500655257905,
                "delta_etabli": 6.25451177375672,
                "delta_bloc_debut": 4.702876243287768,
                "delta_bloc_fin": 4.702876243287768
              },
              "archi2": {
                "mse_d2": 7.299536389112473,
                "mse_all": 6.343453460931778,
                "div_rate": 0.0,
                "delta_d2": -2.3113304360964735,
                "delta_all": -0.7631093936760359,
                "delta_transitoire": 1.040627841821055,
                "delta_etabli": -2.772325938462216,
                "delta_bloc_debut": -2.3173037916788757,
                "delta_bloc_fin": -2.3173037916788757
              }
            }
          }
        },
        "bruit_q_5": {
          "libelle": "q x 5",
          "x": 5.0,
          "T": 160,
          "nominal": false,
          "reference": false,
          "ekf_d2": 7.551122277975082,
          "ekf_all": 8.14456679224968,
          "modeles": {
            "A (etroit)": {
              "archi1": {
                "mse_d2": 29.22715750336647,
                "mse_all": 15.268429204821587,
                "div_rate": 0.0,
                "delta_d2": 5.877750769532017,
                "delta_all": 2.7292637050134116,
                "delta_transitoire": 0.33543466332136673,
                "delta_etabli": 8.476906702184964,
                "delta_bloc_debut": 5.88757174839704,
                "delta_bloc_fin": 5.88757174839704
              },
              "archi2": {
                "mse_d2": 6.774176472425461,
                "mse_all": 6.665700688958168,
                "div_rate": 0.0,
                "delta_d2": -0.4715499685202824,
                "delta_all": -0.8702218066246815,
                "delta_transitoire": 0.6147580673484306,
                "delta_etabli": -0.30736429856827396,
                "delta_bloc_debut": -0.4730708476596617,
                "delta_bloc_fin": -0.4730708476596617
              }
            },
            "B (randomise)": {
              "archi1": {
                "mse_d2": 24.677358163893224,
                "mse_all": 13.628910705447197,
                "div_rate": 0.0,
                "delta_d2": 5.142871614190023,
                "delta_all": 2.23593156883066,
                "delta_transitoire": 0.5204892826029894,
                "delta_etabli": 6.117640326304077,
                "delta_bloc_debut": 5.152061888018497,
                "delta_bloc_fin": 5.152061888018497
              },
              "archi2": {
                "mse_d2": 5.479363113641739,
                "mse_all": 5.748031006753445,
                "div_rate": 0.0,
                "delta_d2": -1.3928142122171316,
                "delta_all": -1.5134888703451725,
                "delta_transitoire": 1.5622082124209005,
                "delta_etabli": -0.7954103109917716,
                "delta_bloc_debut": -1.3978298232476165,
                "delta_bloc_fin": -1.3978298232476165
              }
            }
          }
        },
        "bruit_q_25": {
          "libelle": "q x 25",
          "x": 25.0,
          "T": 160,
          "nominal": false,
          "reference": false,
          "ekf_d2": 31.632534039020538,
          "ekf_all": 39.194954234361646,
          "modeles": {
            "A (etroit)": {
              "archi1": {
                "mse_d2": 9491.944459295273,
                "mse_all": 4079.79171833992,
                "div_rate": 0.15,
                "delta_d2": 24.772212042969727,
                "delta_all": 20.174078303565075,
                "delta_transitoire": 0.20195549483990552,
                "delta_etabli": 27.871589749393245,
                "delta_bloc_debut": 24.77492157375596,
                "delta_bloc_fin": 24.77492157375596
              },
              "archi2": {
                "mse_d2": 135.73825860023499,
                "mse_all": 82.35949547290802,
                "div_rate": 0.0,
                "delta_d2": 6.325682892375833,
                "delta_all": 3.2248351582076458,
                "delta_transitoire": 0.5983832304802801,
                "delta_etabli": 7.721386876179542,
                "delta_bloc_debut": 6.327768085220952,
                "delta_bloc_fin": 6.327768085220952
              }
            },
            "B (randomise)": {
              "archi1": {
                "mse_d2": 113.01148896217346,
                "mse_all": 72.24236872196198,
                "div_rate": 0.0,
                "delta_d2": 5.529886128819883,
                "delta_all": 2.6556181573866446,
                "delta_transitoire": 0.5094482887255829,
                "delta_etabli": 7.174541279163968,
                "delta_bloc_debut": 5.531843844610451,
                "delta_bloc_fin": 5.531843844610451
              },
              "archi2": {
                "mse_d2": 101.86331094503403,
                "mse_all": 107.66706541776657,
                "div_rate": 0.0,
                "delta_d2": 5.078838042460065,
                "delta_all": 4.3885271436550255,
                "delta_transitoire": 1.3122792795048714,
                "delta_etabli": 7.3149172202116635,
                "delta_bloc_debut": 5.0807125784664775,
                "delta_bloc_fin": 5.0807125784664775
              }
            }
          }
        }
      }
    },
    "geometrie": {
      "titre": "Geometrie de la formation (regime de non-linearite de h)",
      "xlabel": "",
      "groupe": "geometrie",
      "scenarios": {
        "geometrie_triangle": {
          "libelle": "triangle (entrainement)",
          "x": null,
          "T": 160,
          "nominal": true,
          "reference": true,
          "ekf_d2": 10.476013921201229,
          "ekf_all": 8.054943853616715,
          "modeles": {
            "A (etroit)": {
              "archi1": {
                "mse_d2": 172.64509480148553,
                "mse_all": 96.4765282601118,
                "div_rate": 0.05,
                "delta_d2": 12.169581767986198,
                "delta_all": 10.783591492975928,
                "delta_transitoire": 1.000545819530882,
                "delta_etabli": 15.14992970243102,
                "delta_bloc_debut": 12.178255582672454,
                "delta_bloc_fin": 12.178255582672454
              },
              "archi2": {
                "mse_d2": 18.169186773896218,
                "mse_all": 11.132915097475053,
                "div_rate": 0.0,
                "delta_d2": 2.391394224881184,
                "delta_all": 1.4054637948905997,
                "delta_transitoire": 0.6008337177100924,
                "delta_etabli": 3.04985599833913,
                "delta_bloc_debut": 2.395306335496145,
                "delta_bloc_fin": 2.395306335496145
              }
            },
            "B (randomise)": {
              "archi1": {
                "mse_d2": 369.19435865283015,
                "mse_all": 197.40971526205539,
                "div_rate": 0.1,
                "delta_d2": 15.47058989445381,
                "delta_all": 13.893060045483159,
                "delta_transitoire": 0.021121710313447388,
                "delta_etabli": 17.99533373233677,
                "delta_bloc_debut": 15.479561755213627,
                "delta_bloc_fin": 15.479561755213627
              },
              "archi2": {
                "mse_d2": 16.888885271549224,
                "mse_all": 9.05324038863182,
                "div_rate": 0.0,
                "delta_d2": 2.074049186142301,
                "delta_all": 0.5074153469753955,
                "delta_transitoire": 1.8691591942846353,
                "delta_etabli": 2.198538661441793,
                "delta_bloc_debut": 2.0775575976430987,
                "delta_bloc_fin": 2.0775575976430987
              }
            }
          }
        },
        "geometrie_ligne": {
          "libelle": "colineaire (d13 = d12 + d23)",
          "x": null,
          "T": 160,
          "nominal": false,
          "reference": false,
          "ekf_d2": 72.99440231025218,
          "ekf_all": 30.26578982770443,
          "modeles": {
            "A (etroit)": {
              "archi1": {
                "mse_d2": 714797.7245620728,
                "mse_all": 767504.8245361329,
                "div_rate": 0.9,
                "delta_d2": 39.908936044858685,
                "delta_all": 44.04129102220905,
                "delta_transitoire": 3.5400968499457774,
                "delta_etabli": 44.47323268891463,
                "delta_bloc_debut": 39.91117690496929,
                "delta_bloc_fin": 39.91117690496929
              },
              "archi2": {
                "mse_d2": 51696.72429971695,
                "mse_all": 66432.98075809478,
                "div_rate": 0.35,
                "delta_d2": 28.501734685754823,
                "delta_all": 33.41431727172918,
                "delta_transitoire": 2.957025719248038,
                "delta_etabli": 32.51171269502224,
                "delta_bloc_debut": 28.503973249097104,
                "delta_bloc_fin": 28.503973249097104
              }
            },
            "B (randomise)": {
              "archi1": {
                "mse_d2": 12054.843150520324,
                "mse_all": 6609.708676242828,
                "div_rate": 0.65,
                "delta_d2": 22.178720071569234,
                "delta_all": 23.392303064100325,
                "delta_transitoire": 4.1274421258233644,
                "delta_etabli": 25.096857453300267,
                "delta_bloc_debut": 22.18094785290012,
                "delta_bloc_fin": 22.18094785290012
              },
              "archi2": {
                "mse_d2": 26152212.702628423,
                "mse_all": 22075287.99123039,
                "div_rate": 0.65,
                "delta_d2": 55.54218882967924,
                "delta_all": 58.62954366121343,
                "delta_transitoire": 1.698532598833506,
                "delta_etabli": 60.15052638303941,
                "delta_bloc_debut": 55.54443000159967,
                "delta_bloc_fin": 55.54443000159967
              }
            }
          }
        },
        "geometrie_serree": {
          "libelle": "formation serree",
          "x": null,
          "T": 160,
          "nominal": false,
          "reference": false,
          "ekf_d2": 31.236337047815322,
          "ekf_all": 29.251264643669128,
          "modeles": {
            "A (etroit)": {
              "archi1": {
                "mse_d2": 24457.15249183178,
                "mse_all": 22329.63297184706,
                "div_rate": 0.4,
                "delta_d2": 28.937457910863632,
                "delta_all": 28.827369376800025,
                "delta_transitoire": -0.8745003312545431,
                "delta_etabli": 33.697836598299936,
                "delta_bloc_debut": 28.94050365622435,
                "delta_bloc_fin": 28.94050365622435
              },
              "archi2": {
                "mse_d2": 47.82357338666916,
                "mse_all": 47.249973964691165,
                "div_rate": 0.0,
                "delta_d2": 1.84981923232808,
                "delta_all": 2.0825692651384964,
                "delta_transitoire": -1.2427444231515459,
                "delta_etabli": 2.5612490881391405,
                "delta_bloc_debut": 1.8508772498798922,
                "delta_bloc_fin": 1.8508772498798922
              }
            },
            "B (randomise)": {
              "archi1": {
                "mse_d2": 676.7786357045173,
                "mse_all": 340.01870665550234,
                "div_rate": 0.05,
                "delta_d2": 13.357865402739769,
                "delta_all": 10.653581640527932,
                "delta_transitoire": -0.46992698489091667,
                "delta_etabli": 15.96043865345267,
                "delta_bloc_debut": 13.360774025445412,
                "delta_bloc_fin": 13.360774025445412
              },
              "archi2": {
                "mse_d2": 63.72053759694099,
                "mse_all": 65.51882287859917,
                "div_rate": 0.0,
                "delta_d2": 3.0961933086969537,
                "delta_all": 3.502214391900172,
                "delta_transitoire": -1.7859218724315251,
                "delta_etabli": 5.321002525731875,
                "delta_bloc_debut": 3.0977481781726146,
                "delta_bloc_fin": 3.0977481781726146
              }
            }
          }
        },
        "geometrie_large": {
          "libelle": "formation large",
          "x": null,
          "T": 160,
          "nominal": false,
          "reference": false,
          "ekf_d2": 6.394897454977036,
          "ekf_all": 6.088793477416038,
          "modeles": {
            "A (etroit)": {
              "archi1": {
                "mse_d2": 5.810346269607544,
                "mse_all": 5.6530058398842815,
                "div_rate": 0.0,
                "delta_d2": -0.4163156967312793,
                "delta_all": -0.3225180958634865,
                "delta_transitoire": 0.4491560268595974,
                "delta_etabli": -1.7186087184576522,
                "delta_bloc_debut": -0.4180460516421436,
                "delta_bloc_fin": -0.4180460516421436
              },
              "archi2": {
                "mse_d2": 7.3700287260115145,
                "mse_all": 7.369129510968923,
                "div_rate": 0.0,
                "delta_d2": 0.6163559583768106,
                "delta_all": 0.8288494552983463,
                "delta_transitoire": 0.828561761587795,
                "delta_etabli": -0.02911367127996628,
                "delta_bloc_debut": 0.6186301023059646,
                "delta_bloc_fin": 0.6186301023059646
              }
            },
            "B (randomise)": {
              "archi1": {
                "mse_d2": 7.395592805743218,
                "mse_all": 7.056484478712082,
                "div_rate": 0.0,
                "delta_d2": 0.6313940654169834,
                "delta_all": 0.6405714690744747,
                "delta_transitoire": 0.06620562190947728,
                "delta_etabli": -0.18167660439782737,
                "delta_bloc_debut": 0.6337197385946132,
                "delta_bloc_fin": 0.6337197385946132
              },
              "archi2": {
                "mse_d2": 7.365890502184629,
                "mse_all": 7.5828797273337845,
                "div_rate": 0.0,
                "delta_d2": 0.6139167378676447,
                "delta_all": 0.9530292396323364,
                "delta_transitoire": 0.8122833347933651,
                "delta_etabli": 0.0680457607377123,
                "delta_bloc_debut": 0.6161825695579769,
                "delta_bloc_fin": 0.6161825695579769
              }
            }
          }
        }
      }
    },
    "condition_initiale": {
      "titre": "Erreur sur l'etat initial (les filtres partent toujours de x0 nominal)",
      "xlabel": "Amplitude de la perturbation initiale (x chol(P0))",
      "groupe": "condition_initiale",
      "scenarios": {
        "condition_initiale_0": {
          "libelle": "offset x 0",
          "x": 0.0,
          "T": 160,
          "nominal": false,
          "reference": false,
          "ekf_d2": 4.221273398399353,
          "ekf_all": 4.007446968555451,
          "modeles": {
            "A (etroit)": {
              "archi1": {
                "mse_d2": 0.2708297915756702,
                "mse_all": 0.2000537022948265,
                "div_rate": 0.0,
                "delta_d2": -11.927470454195362,
                "delta_all": -13.017211908855678,
                "delta_transitoire": -1.8861883216786957,
                "delta_etabli": -14.917822884014893,
                "delta_bloc_debut": -11.92747063570845,
                "delta_bloc_fin": -11.92747063570845
              },
              "archi2": {
                "mse_d2": 0.7044760063290596,
                "mse_all": 0.49158416464924815,
                "div_rate": 0.0,
                "delta_d2": -7.77577274823007,
                "delta_all": -9.112698993922065,
                "delta_transitoire": -4.309312176427212,
                "delta_etabli": -8.790469323502476,
                "delta_bloc_debut": -7.775772967961483,
                "delta_bloc_fin": -7.775772967961483
              }
            },
            "B (randomise)": {
              "archi1": {
                "mse_d2": 0.565654294192791,
                "mse_all": 0.4663471892476082,
                "div_rate": 0.0,
                "delta_d2": -8.728923924901608,
                "delta_all": -9.341584204065338,
                "delta_transitoire": -2.242067349733421,
                "delta_etabli": -9.63513295519317,
                "delta_bloc_debut": -8.728924026617227,
                "delta_bloc_fin": -8.728924026617227
              },
              "archi2": {
                "mse_d2": 0.38057988360524175,
                "mse_all": 0.21286240965127945,
                "div_rate": 0.0,
                "delta_d2": -10.449976519736504,
                "delta_all": -12.747688097919653,
                "delta_transitoire": -4.3545187258936195,
                "delta_etabli": -14.991176999113447,
                "delta_bloc_debut": -10.44997669535212,
                "delta_bloc_fin": -10.44997669535212
              }
            }
          }
        },
        "condition_initiale_0.5": {
          "libelle": "offset x 0.5",
          "x": 0.5,
          "T": 160,
          "nominal": false,
          "reference": false,
          "ekf_d2": 3.663573516905308,
          "ekf_all": 3.5949228167533875,
          "modeles": {
            "A (etroit)": {
              "archi1": {
                "mse_d2": 2.7237085327506065,
                "mse_all": 2.5480832643806934,
                "div_rate": 0.0,
                "delta_d2": -1.2874427958236478,
                "delta_all": -1.4947595503669504,
                "delta_transitoire": -0.3114313936864095,
                "delta_etabli": -1.1911175104853995,
                "delta_bloc_debut": -1.2891622442906092,
                "delta_bloc_fin": -1.2891622442906092
              },
              "archi2": {
                "mse_d2": 1.6741521947085858,
                "mse_all": 1.7078103624284267,
                "div_rate": 0.0,
                "delta_d2": -3.4010997442373503,
                "delta_all": -3.2324992610199317,
                "delta_transitoire": -0.6820508383744366,
                "delta_etabli": -3.548992167666074,
                "delta_bloc_debut": -3.407023783437829,
                "delta_bloc_fin": -3.407023783437829
              }
            },
            "B (randomise)": {
              "archi1": {
                "mse_d2": 2.713986712694168,
                "mse_all": 2.8053172409534453,
                "div_rate": 0.0,
                "delta_d2": -1.3029719386409087,
                "delta_all": -1.0770758966904466,
                "delta_transitoire": -0.17684057568141814,
                "delta_etabli": -1.3195502847364855,
                "delta_bloc_debut": -1.3047154984733231,
                "delta_bloc_fin": -1.3047154984733231
              },
              "archi2": {
                "mse_d2": 1.3554970644414426,
                "mse_all": 1.3580080434679984,
                "div_rate": 0.0,
                "delta_d2": -4.318063295676976,
                "delta_all": -4.227872282118631,
                "delta_transitoire": 0.5477222605012985,
                "delta_etabli": -5.325329876304199,
                "delta_bloc_debut": -4.32655469203657,
                "delta_bloc_fin": -4.32655469203657
              }
            }
          }
        },
        "condition_initiale_1": {
          "libelle": "offset x 1",
          "x": 1.0,
          "T": 160,
          "nominal": true,
          "reference": true,
          "ekf_d2": 10.271346279978752,
          "ekf_all": 8.583713141083717,
          "modeles": {
            "A (etroit)": {
              "archi1": {
                "mse_d2": 25.97130459100008,
                "mse_all": 15.487572263926268,
                "div_rate": 0.05,
                "delta_d2": 4.028663947406907,
                "delta_all": 2.563081502535607,
                "delta_transitoire": 0.7358131232561343,
                "delta_etabli": 4.816276024236235,
                "delta_bloc_debut": 4.033791346641822,
                "delta_bloc_fin": 4.033791346641822
              },
              "archi2": {
                "mse_d2": 14.708174746483564,
                "mse_all": 10.471665333211423,
                "div_rate": 0.0,
                "delta_d2": 1.559314101165136,
                "delta_all": 0.8634055850662581,
                "delta_transitoire": 1.503325507654298,
                "delta_etabli": 2.3754890697990425,
                "delta_bloc_debut": 1.5618733751529739,
                "delta_bloc_fin": 1.5618733751529739
              }
            },
            "B (randomise)": {
              "archi1": {
                "mse_d2": 15.969111779332161,
                "mse_all": 11.748543670773506,
                "div_rate": 0.0,
                "delta_d2": 1.9165338991928216,
                "delta_all": 1.3630884002566797,
                "delta_transitoire": 0.23800255913577334,
                "delta_etabli": 2.2233311223932724,
                "delta_bloc_debut": 1.9195608837670424,
                "delta_bloc_fin": 1.9195608837670424
              },
              "archi2": {
                "mse_d2": 15.558902856707572,
                "mse_all": 10.838397145271301,
                "div_rate": 0.0,
                "delta_d2": 1.8035159833128183,
                "delta_all": 1.0128986505270798,
                "delta_transitoire": 2.1939997481036544,
                "delta_etabli": 2.605932239839266,
                "delta_bloc_debut": 1.806399279417774,
                "delta_bloc_fin": 1.806399279417774
              }
            }
          }
        },
        "condition_initiale_2": {
          "libelle": "offset x 2",
          "x": 2.0,
          "T": 160,
          "nominal": false,
          "reference": false,
          "ekf_d2": 45.4243870139122,
          "ekf_all": 36.71119617819786,
          "modeles": {
            "A (etroit)": {
              "archi1": {
                "mse_d2": 6981.266296863556,
                "mse_all": 5152.720979452133,
                "div_rate": 0.05,
                "delta_d2": 21.86645128896487,
                "delta_all": 21.472380904357983,
                "delta_transitoire": 0.5397053164325715,
                "delta_etabli": 25.78960161763339,
                "delta_bloc_debut": 21.87675242986581,
                "delta_bloc_fin": 21.87675242986581
              },
              "archi2": {
                "mse_d2": 88.86180274486541,
                "mse_all": 78.36566898822784,
                "div_rate": 0.0,
                "delta_d2": 2.9142604418442857,
                "delta_all": 3.2932730985290934,
                "delta_transitoire": -0.24420662328847983,
                "delta_etabli": 3.517679713039772,
                "delta_bloc_debut": 2.9193321111253994,
                "delta_bloc_fin": 2.9193321111253994
              }
            },
            "B (randomise)": {
              "archi1": {
                "mse_d2": 905.2607051849366,
                "mse_all": 482.9346987843513,
                "div_rate": 0.05,
                "delta_d2": 12.994845939118518,
                "delta_all": 11.19089874957387,
                "delta_transitoire": -0.230838635209475,
                "delta_etabli": 15.395174765836035,
                "delta_bloc_debut": 13.004695320457051,
                "delta_bloc_fin": 13.004695320457051
              },
              "archi2": {
                "mse_d2": 117.87629903554917,
                "mse_all": 73.6333112001419,
                "div_rate": 0.0,
                "delta_d2": 4.141374164709573,
                "delta_all": 3.022757950524511,
                "delta_transitoire": 0.6154569097803322,
                "delta_etabli": 4.6672987133655,
                "delta_bloc_debut": 4.1477503334075685,
                "delta_bloc_fin": 4.1477503334075685
              }
            }
          }
        },
        "condition_initiale_3": {
          "libelle": "offset x 3",
          "x": 3.0,
          "T": 160,
          "nominal": false,
          "reference": false,
          "ekf_d2": 435.1112651824951,
          "ekf_all": 187.9531417965889,
          "modeles": {
            "A (etroit)": {
              "archi1": {
                "mse_d2": 2351.295694255829,
                "mse_all": 1386.7026914596559,
                "div_rate": 0.2,
                "delta_d2": 7.32706921093116,
                "delta_all": 8.679337687663438,
                "delta_transitoire": -0.08124435481521206,
                "delta_etabli": 10.540000142358867,
                "delta_bloc_debut": 7.328776074615958,
                "delta_bloc_fin": 7.328776074615958
              },
              "archi2": {
                "mse_d2": 480.1401304244995,
                "mse_all": 217.72430543899537,
                "div_rate": 0.05,
                "delta_d2": 0.4276767862758183,
                "delta_all": 0.6385732407328487,
                "delta_transitoire": -0.0556461045100269,
                "delta_etabli": 0.7620605738995045,
                "delta_bloc_debut": 0.42787361663236345,
                "delta_bloc_fin": 0.42787361663236345
              }
            },
            "B (randomise)": {
              "archi1": {
                "mse_d2": 1461.4654476165772,
                "mse_all": 657.5531750202179,
                "div_rate": 0.15,
                "delta_d2": 5.261882246267123,
                "delta_all": 5.438812894026737,
                "delta_transitoire": -0.6719247149538498,
                "delta_etabli": 6.304375301979357,
                "delta_bloc_debut": 5.263353053637781,
                "delta_bloc_fin": 5.263353053637781
              },
              "archi2": {
                "mse_d2": 374.2313796520233,
                "mse_all": 216.37561383247376,
                "div_rate": 0.05,
                "delta_d2": -0.6546012678367634,
                "delta_all": 0.611587232161089,
                "delta_transitoire": 0.30777207117933797,
                "delta_etabli": -0.038139037099904946,
                "delta_bloc_debut": -0.6549419977440518,
                "delta_bloc_fin": -0.6549419977440518
              }
            }
          }
        },
        "condition_initiale_5": {
          "libelle": "offset x 5",
          "x": 5.0,
          "T": 160,
          "nominal": false,
          "reference": false,
          "ekf_d2": 769.3877686500549,
          "ekf_all": 358.0105929851532,
          "modeles": {
            "A (etroit)": {
              "archi1": {
                "mse_d2": 9114.944691085815,
                "mse_all": 9457.889820671082,
                "div_rate": 0.3,
                "delta_d2": 10.736087599025478,
                "delta_all": 14.218983734691903,
                "delta_transitoire": -0.44763320948546986,
                "delta_etabli": 14.208268045156656,
                "delta_bloc_debut": 10.73892365601056,
                "delta_bloc_fin": 10.73892365601056
              },
              "archi2": {
                "mse_d2": 1627.5347317695619,
                "mse_all": 2127.5647954463957,
                "div_rate": 0.0,
                "delta_d2": 3.2538498740626998,
                "delta_all": 7.739869185479122,
                "delta_transitoire": -1.5481318580846466,
                "delta_etabli": 4.326545683402601,
                "delta_bloc_debut": 3.2554832373812905,
                "delta_bloc_fin": 3.2554832373812905
              }
            },
            "B (randomise)": {
              "archi1": {
                "mse_d2": 16662.42683086395,
                "mse_all": 9158.679846000672,
                "div_rate": 0.4,
                "delta_d2": 13.355929774991385,
                "delta_all": 14.079370009920613,
                "delta_transitoire": -1.812138041884375,
                "delta_etabli": 15.843866326470986,
                "delta_bloc_debut": 13.35888408202512,
                "delta_bloc_fin": 13.35888408202512
              },
              "archi2": {
                "mse_d2": 2970606.087092495,
                "mse_all": 2916111.581810379,
                "div_rate": 0.2,
                "delta_d2": 35.86699788807613,
                "delta_all": 39.10908260823174,
                "delta_transitoire": -1.0199681323903012,
                "delta_etabli": 40.35310681479882,
                "delta_bloc_debut": 35.87009430657061,
                "delta_bloc_fin": 35.87009430657061
              }
            }
          }
        }
      }
    }
  }
}
