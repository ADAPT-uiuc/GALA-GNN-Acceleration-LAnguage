/* A Bison parser, made by GNU Bison 3.8.2.  */

/* Bison implementation for Yacc-like parsers in C

   Copyright (C) 1984, 1989-1990, 2000-2015, 2018-2021 Free Software Foundation,
   Inc.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <https://www.gnu.org/licenses/>.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.

   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */

/* C LALR(1) parser skeleton written by Richard Stallman, by
   simplifying the original so-called "semantic" parser.  */

/* DO NOT RELY ON FEATURES THAT ARE NOT DOCUMENTED in the manual,
   especially those whose name start with YY_ or yy_.  They are
   private implementation details that can be changed or removed.  */

/* All symbols defined below should begin with yy or YY, to avoid
   infringing on user name space.  This should be done even for local
   variables, as they might otherwise be expanded by user macros.
   There are some unavoidable exceptions within include files to
   define necessary library symbols; they are noted "INFRINGES ON
   USER NAME SPACE" below.  */

/* Identify Bison output, and Bison version.  */
#define YYBISON 30802

/* Bison version string.  */
#define YYBISON_VERSION "3.8.2"

/* Skeleton name.  */
#define YYSKELETON_NAME "yacc.c"

/* Pure parsers.  */
#define YYPURE 0

/* Push parsers.  */
#define YYPUSH 0

/* Pull parsers.  */
#define YYPULL 1




/* First part of user prologue.  */
#line 1 "frontend.y"

#include <iostream>
#include <string.h>
#include <vector>
#include <map>
#include "ir/data.h"
#include "ir/compute.h"
#include "ir/frontend_metadata.h"
using namespace std;

extern int yydebug;
extern int yylex();
extern int yyparse();
extern FILE *yyin;

void yyerror(const char *s);

ModelConfig m1 = ModelConfig(); // just considering one model per input file for now
int debug = 0;

DataNode* normData; // very temp solution, find fix asap

vector<CIRNode*> program;
vector<RelationEdge*> dependencies;
vector<RelationEdge*> associations;
vector<TransformEdge*> transforms;

#line 99 "build/frontend.tab.c"

# ifndef YY_CAST
#  ifdef __cplusplus
#   define YY_CAST(Type, Val) static_cast<Type> (Val)
#   define YY_REINTERPRET_CAST(Type, Val) reinterpret_cast<Type> (Val)
#  else
#   define YY_CAST(Type, Val) ((Type) (Val))
#   define YY_REINTERPRET_CAST(Type, Val) ((Type) (Val))
#  endif
# endif
# ifndef YY_NULLPTR
#  if defined __cplusplus
#   if 201103L <= __cplusplus
#    define YY_NULLPTR nullptr
#   else
#    define YY_NULLPTR 0
#   endif
#  else
#   define YY_NULLPTR ((void*)0)
#  endif
# endif

#include "frontend.tab.h"
/* Symbol kind.  */
enum yysymbol_kind_t
{
  YYSYMBOL_YYEMPTY = -2,
  YYSYMBOL_YYEOF = 0,                      /* "end of file"  */
  YYSYMBOL_YYerror = 1,                    /* error  */
  YYSYMBOL_YYUNDEF = 2,                    /* "invalid token"  */
  YYSYMBOL_IDENTIFIER = 3,                 /* IDENTIFIER  */
  YYSYMBOL_ASSIGN = 4,                     /* ASSIGN  */
  YYSYMBOL_LOAD = 5,                       /* LOAD  */
  YYSYMBOL_LPAREN = 6,                     /* LPAREN  */
  YYSYMBOL_RPAREN = 7,                     /* RPAREN  */
  YYSYMBOL_SEMICOLON = 8,                  /* SEMICOLON  */
  YYSYMBOL_QUOTE = 9,                      /* QUOTE  */
  YYSYMBOL_SET_UNWEIGHTED = 10,            /* SET_UNWEIGHTED  */
  YYSYMBOL_SET_UNDIRECTED = 11,            /* SET_UNDIRECTED  */
  YYSYMBOL_MODEL_W = 12,                   /* MODEL_W  */
  YYSYMBOL_EVAL = 13,                      /* EVAL  */
  YYSYMBOL_TRAIN = 14,                     /* TRAIN  */
  YYSYMBOL_LAYER = 15,                     /* LAYER  */
  YYSYMBOL_ITERS = 16,                     /* ITERS  */
  YYSYMBOL_VAL_STEP = 17,                  /* VAL_STEP  */
  YYSYMBOL_AGGR_INIT = 18,                 /* AGGR_INIT  */
  YYSYMBOL_FN_ARG = 19,                    /* FN_ARG  */
  YYSYMBOL_MUL_SUM = 20,                   /* MUL_SUM  */
  YYSYMBOL_DSL_DOT = 21,                   /* DSL_DOT  */
  YYSYMBOL_FFN_OUT = 22,                   /* FFN_OUT  */
  YYSYMBOL_SIZE_FN = 23,                   /* SIZE_FN  */
  YYSYMBOL_GRAPH_ATTR = 24,                /* GRAPH_ATTR  */
  YYSYMBOL_FEAT_ATTR = 25,                 /* FEAT_ATTR  */
  YYSYMBOL_RELU = 26,                      /* RELU  */
  YYSYMBOL_LABEL_ATTR = 27,                /* LABEL_ATTR  */
  YYSYMBOL_DEGREE_ATTR = 28,               /* DEGREE_ATTR  */
  YYSYMBOL_NODE_ATTR = 29,                 /* NODE_ATTR  */
  YYSYMBOL_LEAKY_RELU = 30,                /* LEAKY_RELU  */
  YYSYMBOL_POW = 31,                       /* POW  */
  YYSYMBOL_SCALAR_INIT = 32,               /* SCALAR_INIT  */
  YYSYMBOL_COLTILE = 33,                   /* COLTILE  */
  YYSYMBOL_FEAT_SIZE_ASSIGN = 34,          /* FEAT_SIZE_ASSIGN  */
  YYSYMBOL_LABEL_SIZE_ASSIGN = 35,         /* LABEL_SIZE_ASSIGN  */
  YYSYMBOL_COARSEN = 36,                   /* COARSEN  */
  YYSYMBOL_SRC_ATTR = 37,                  /* SRC_ATTR  */
  YYSYMBOL_DST_ATTR = 38,                  /* DST_ATTR  */
  YYSYMBOL_INTEGER = 39,                   /* INTEGER  */
  YYSYMBOL_FLOAT = 40,                     /* FLOAT  */
  YYSYMBOL_SOFTMAX = 41,                   /* SOFTMAX  */
  YYSYMBOL_INIT_WEIGHT = 42,               /* INIT_WEIGHT  */
  YYSYMBOL_LBRACE = 43,                    /* LBRACE  */
  YYSYMBOL_RBRACE = 44,                    /* RBRACE  */
  YYSYMBOL_DOT = 45,                       /* DOT  */
  YYSYMBOL_COMMA = 46,                     /* COMMA  */
  YYSYMBOL_TR = 47,                        /* TR  */
  YYSYMBOL_FA = 48,                        /* FA  */
  YYSYMBOL_PLUS = 49,                      /* PLUS  */
  YYSYMBOL_MINUS = 50,                     /* MINUS  */
  YYSYMBOL_MULTIPLY = 51,                  /* MULTIPLY  */
  YYSYMBOL_DIVIDE = 52,                    /* DIVIDE  */
  YYSYMBOL_FFN = 53,                       /* FFN  */
  YYSYMBOL_NULL_KEY = 54,                  /* NULL_KEY  */
  YYSYMBOL_LOSS = 55,                      /* LOSS  */
  YYSYMBOL_OPTIMIZER = 56,                 /* OPTIMIZER  */
  YYSYMBOL_RMSE_LOSS = 57,                 /* RMSE_LOSS  */
  YYSYMBOL_ADAM_T = 58,                    /* ADAM_T  */
  YYSYMBOL_DSL_FN = 59,                    /* DSL_FN  */
  YYSYMBOL_RELAXNLN = 60,                  /* RELAXNLN  */
  YYSYMBOL_QUANT = 61,                     /* QUANT  */
  YYSYMBOL_RABBIT_REORDER_OP = 62,         /* RABBIT_REORDER_OP  */
  YYSYMBOL_SAMPLE_RANDOM_OP = 63,          /* SAMPLE_RANDOM_OP  */
  YYSYMBOL_AGGR = 64,                      /* AGGR  */
  YYSYMBOL_LSQBRA = 65,                    /* LSQBRA  */
  YYSYMBOL_RSQBRA = 66,                    /* RSQBRA  */
  YYSYMBOL_IF = 67,                        /* IF  */
  YYSYMBOL_ELSE = 68,                      /* ELSE  */
  YYSYMBOL_DO = 69,                        /* DO  */
  YYSYMBOL_WHILE = 70,                     /* WHILE  */
  YYSYMBOL_NOT = 71,                       /* NOT  */
  YYSYMBOL_AND = 72,                       /* AND  */
  YYSYMBOL_OR = 73,                        /* OR  */
  YYSYMBOL_NOTEQ = 74,                     /* NOTEQ  */
  YYSYMBOL_EQ = 75,                        /* EQ  */
  YYSYMBOL_GREATER = 76,                   /* GREATER  */
  YYSYMBOL_LESS = 77,                      /* LESS  */
  YYSYMBOL_GREATEREQ = 78,                 /* GREATEREQ  */
  YYSYMBOL_LESSEQ = 79,                    /* LESSEQ  */
  YYSYMBOL_DATASET = 80,                   /* DATASET  */
  YYSYMBOL_NONLN = 81,                     /* NONLN  */
  YYSYMBOL_SENSEI_OP = 82,                 /* SENSEI_OP  */
  YYSYMBOL_INT = 83,                       /* INT  */
  YYSYMBOL_NEW = 84,                       /* NEW  */
  YYSYMBOL_YYACCEPT = 85,                  /* $accept  */
  YYSYMBOL_program = 86,                   /* program  */
  YYSYMBOL_load_dataset = 87,              /* load_dataset  */
  YYSYMBOL_algorithm = 88,                 /* algorithm  */
  YYSYMBOL_statements = 89,                /* statements  */
  YYSYMBOL_statement = 90,                 /* statement  */
  YYSYMBOL_layers = 91,                    /* layers  */
  YYSYMBOL_layer_def = 92,                 /* layer_def  */
  YYSYMBOL_model = 93,                     /* model  */
  YYSYMBOL_model_def = 94,                 /* model_def  */
  YYSYMBOL_layer_inits = 95,               /* layer_inits  */
  YYSYMBOL_layer_init = 96,                /* layer_init  */
  YYSYMBOL_model_init = 97,                /* model_init  */
  YYSYMBOL_model_uses = 98,                /* model_uses  */
  YYSYMBOL_model_use = 99,                 /* model_use  */
  YYSYMBOL_gnn_op = 100,                   /* gnn_op  */
  YYSYMBOL_function = 101,                 /* function  */
  YYSYMBOL_update_op = 102,                /* update_op  */
  YYSYMBOL_schedules = 103,                /* schedules  */
  YYSYMBOL_schedule = 104,                 /* schedule  */
  YYSYMBOL_data_transform = 105,           /* data_transform  */
  YYSYMBOL_function_transform = 106,       /* function_transform  */
  YYSYMBOL_data_var = 107,                 /* data_var  */
  YYSYMBOL_function_init = 108,            /* function_init  */
  YYSYMBOL_semiring_op = 109,              /* semiring_op  */
  YYSYMBOL_op = 110,                       /* op  */
  YYSYMBOL_train_args = 111,               /* train_args  */
  YYSYMBOL_train_arg = 112,                /* train_arg  */
  YYSYMBOL_args = 113,                     /* args  */
  YYSYMBOL_arg = 114,                      /* arg  */
  YYSYMBOL_bool = 115,                     /* bool  */
  YYSYMBOL_string = 116                    /* string  */
};
typedef enum yysymbol_kind_t yysymbol_kind_t;




#ifdef short
# undef short
#endif

/* On compilers that do not define __PTRDIFF_MAX__ etc., make sure
   <limits.h> and (if available) <stdint.h> are included
   so that the code can choose integer types of a good width.  */

#ifndef __PTRDIFF_MAX__
# include <limits.h> /* INFRINGES ON USER NAME SPACE */
# if defined __STDC_VERSION__ && 199901 <= __STDC_VERSION__
#  include <stdint.h> /* INFRINGES ON USER NAME SPACE */
#  define YY_STDINT_H
# endif
#endif

/* Narrow types that promote to a signed type and that can represent a
   signed or unsigned integer of at least N bits.  In tables they can
   save space and decrease cache pressure.  Promoting to a signed type
   helps avoid bugs in integer arithmetic.  */

#ifdef __INT_LEAST8_MAX__
typedef __INT_LEAST8_TYPE__ yytype_int8;
#elif defined YY_STDINT_H
typedef int_least8_t yytype_int8;
#else
typedef signed char yytype_int8;
#endif

#ifdef __INT_LEAST16_MAX__
typedef __INT_LEAST16_TYPE__ yytype_int16;
#elif defined YY_STDINT_H
typedef int_least16_t yytype_int16;
#else
typedef short yytype_int16;
#endif

/* Work around bug in HP-UX 11.23, which defines these macros
   incorrectly for preprocessor constants.  This workaround can likely
   be removed in 2023, as HPE has promised support for HP-UX 11.23
   (aka HP-UX 11i v2) only through the end of 2022; see Table 2 of
   <https://h20195.www2.hpe.com/V2/getpdf.aspx/4AA4-7673ENW.pdf>.  */
#ifdef __hpux
# undef UINT_LEAST8_MAX
# undef UINT_LEAST16_MAX
# define UINT_LEAST8_MAX 255
# define UINT_LEAST16_MAX 65535
#endif

#if defined __UINT_LEAST8_MAX__ && __UINT_LEAST8_MAX__ <= __INT_MAX__
typedef __UINT_LEAST8_TYPE__ yytype_uint8;
#elif (!defined __UINT_LEAST8_MAX__ && defined YY_STDINT_H \
       && UINT_LEAST8_MAX <= INT_MAX)
typedef uint_least8_t yytype_uint8;
#elif !defined __UINT_LEAST8_MAX__ && UCHAR_MAX <= INT_MAX
typedef unsigned char yytype_uint8;
#else
typedef short yytype_uint8;
#endif

#if defined __UINT_LEAST16_MAX__ && __UINT_LEAST16_MAX__ <= __INT_MAX__
typedef __UINT_LEAST16_TYPE__ yytype_uint16;
#elif (!defined __UINT_LEAST16_MAX__ && defined YY_STDINT_H \
       && UINT_LEAST16_MAX <= INT_MAX)
typedef uint_least16_t yytype_uint16;
#elif !defined __UINT_LEAST16_MAX__ && USHRT_MAX <= INT_MAX
typedef unsigned short yytype_uint16;
#else
typedef int yytype_uint16;
#endif

#ifndef YYPTRDIFF_T
# if defined __PTRDIFF_TYPE__ && defined __PTRDIFF_MAX__
#  define YYPTRDIFF_T __PTRDIFF_TYPE__
#  define YYPTRDIFF_MAXIMUM __PTRDIFF_MAX__
# elif defined PTRDIFF_MAX
#  ifndef ptrdiff_t
#   include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  endif
#  define YYPTRDIFF_T ptrdiff_t
#  define YYPTRDIFF_MAXIMUM PTRDIFF_MAX
# else
#  define YYPTRDIFF_T long
#  define YYPTRDIFF_MAXIMUM LONG_MAX
# endif
#endif

#ifndef YYSIZE_T
# ifdef __SIZE_TYPE__
#  define YYSIZE_T __SIZE_TYPE__
# elif defined size_t
#  define YYSIZE_T size_t
# elif defined __STDC_VERSION__ && 199901 <= __STDC_VERSION__
#  include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  define YYSIZE_T size_t
# else
#  define YYSIZE_T unsigned
# endif
#endif

#define YYSIZE_MAXIMUM                                  \
  YY_CAST (YYPTRDIFF_T,                                 \
           (YYPTRDIFF_MAXIMUM < YY_CAST (YYSIZE_T, -1)  \
            ? YYPTRDIFF_MAXIMUM                         \
            : YY_CAST (YYSIZE_T, -1)))

#define YYSIZEOF(X) YY_CAST (YYPTRDIFF_T, sizeof (X))


/* Stored state numbers (used for stacks). */
typedef yytype_uint8 yy_state_t;

/* State numbers in computations.  */
typedef int yy_state_fast_t;

#ifndef YY_
# if defined YYENABLE_NLS && YYENABLE_NLS
#  if ENABLE_NLS
#   include <libintl.h> /* INFRINGES ON USER NAME SPACE */
#   define YY_(Msgid) dgettext ("bison-runtime", Msgid)
#  endif
# endif
# ifndef YY_
#  define YY_(Msgid) Msgid
# endif
#endif


#ifndef YY_ATTRIBUTE_PURE
# if defined __GNUC__ && 2 < __GNUC__ + (96 <= __GNUC_MINOR__)
#  define YY_ATTRIBUTE_PURE __attribute__ ((__pure__))
# else
#  define YY_ATTRIBUTE_PURE
# endif
#endif

#ifndef YY_ATTRIBUTE_UNUSED
# if defined __GNUC__ && 2 < __GNUC__ + (7 <= __GNUC_MINOR__)
#  define YY_ATTRIBUTE_UNUSED __attribute__ ((__unused__))
# else
#  define YY_ATTRIBUTE_UNUSED
# endif
#endif

/* Suppress unused-variable warnings by "using" E.  */
#if ! defined lint || defined __GNUC__
# define YY_USE(E) ((void) (E))
#else
# define YY_USE(E) /* empty */
#endif

/* Suppress an incorrect diagnostic about yylval being uninitialized.  */
#if defined __GNUC__ && ! defined __ICC && 406 <= __GNUC__ * 100 + __GNUC_MINOR__
# if __GNUC__ * 100 + __GNUC_MINOR__ < 407
#  define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN                           \
    _Pragma ("GCC diagnostic push")                                     \
    _Pragma ("GCC diagnostic ignored \"-Wuninitialized\"")
# else
#  define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN                           \
    _Pragma ("GCC diagnostic push")                                     \
    _Pragma ("GCC diagnostic ignored \"-Wuninitialized\"")              \
    _Pragma ("GCC diagnostic ignored \"-Wmaybe-uninitialized\"")
# endif
# define YY_IGNORE_MAYBE_UNINITIALIZED_END      \
    _Pragma ("GCC diagnostic pop")
#else
# define YY_INITIAL_VALUE(Value) Value
#endif
#ifndef YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_END
#endif
#ifndef YY_INITIAL_VALUE
# define YY_INITIAL_VALUE(Value) /* Nothing. */
#endif

#if defined __cplusplus && defined __GNUC__ && ! defined __ICC && 6 <= __GNUC__
# define YY_IGNORE_USELESS_CAST_BEGIN                          \
    _Pragma ("GCC diagnostic push")                            \
    _Pragma ("GCC diagnostic ignored \"-Wuseless-cast\"")
# define YY_IGNORE_USELESS_CAST_END            \
    _Pragma ("GCC diagnostic pop")
#endif
#ifndef YY_IGNORE_USELESS_CAST_BEGIN
# define YY_IGNORE_USELESS_CAST_BEGIN
# define YY_IGNORE_USELESS_CAST_END
#endif


#define YY_ASSERT(E) ((void) (0 && (E)))

#if !defined yyoverflow

/* The parser invokes alloca or malloc; define the necessary symbols.  */

# ifdef YYSTACK_USE_ALLOCA
#  if YYSTACK_USE_ALLOCA
#   ifdef __GNUC__
#    define YYSTACK_ALLOC __builtin_alloca
#   elif defined __BUILTIN_VA_ARG_INCR
#    include <alloca.h> /* INFRINGES ON USER NAME SPACE */
#   elif defined _AIX
#    define YYSTACK_ALLOC __alloca
#   elif defined _MSC_VER
#    include <malloc.h> /* INFRINGES ON USER NAME SPACE */
#    define alloca _alloca
#   else
#    define YYSTACK_ALLOC alloca
#    if ! defined _ALLOCA_H && ! defined EXIT_SUCCESS
#     include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
      /* Use EXIT_SUCCESS as a witness for stdlib.h.  */
#     ifndef EXIT_SUCCESS
#      define EXIT_SUCCESS 0
#     endif
#    endif
#   endif
#  endif
# endif

# ifdef YYSTACK_ALLOC
   /* Pacify GCC's 'empty if-body' warning.  */
#  define YYSTACK_FREE(Ptr) do { /* empty */; } while (0)
#  ifndef YYSTACK_ALLOC_MAXIMUM
    /* The OS might guarantee only one guard page at the bottom of the stack,
       and a page size can be as small as 4096 bytes.  So we cannot safely
       invoke alloca (N) if N exceeds 4096.  Use a slightly smaller number
       to allow for a few compiler-allocated temporary stack slots.  */
#   define YYSTACK_ALLOC_MAXIMUM 4032 /* reasonable circa 2006 */
#  endif
# else
#  define YYSTACK_ALLOC YYMALLOC
#  define YYSTACK_FREE YYFREE
#  ifndef YYSTACK_ALLOC_MAXIMUM
#   define YYSTACK_ALLOC_MAXIMUM YYSIZE_MAXIMUM
#  endif
#  if (defined __cplusplus && ! defined EXIT_SUCCESS \
       && ! ((defined YYMALLOC || defined malloc) \
             && (defined YYFREE || defined free)))
#   include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#   ifndef EXIT_SUCCESS
#    define EXIT_SUCCESS 0
#   endif
#  endif
#  ifndef YYMALLOC
#   define YYMALLOC malloc
#   if ! defined malloc && ! defined EXIT_SUCCESS
void *malloc (YYSIZE_T); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
#  ifndef YYFREE
#   define YYFREE free
#   if ! defined free && ! defined EXIT_SUCCESS
void free (void *); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
# endif
#endif /* !defined yyoverflow */

#if (! defined yyoverflow \
     && (! defined __cplusplus \
         || (defined YYSTYPE_IS_TRIVIAL && YYSTYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union yyalloc
{
  yy_state_t yyss_alloc;
  YYSTYPE yyvs_alloc;
};

/* The size of the maximum gap between one aligned stack and the next.  */
# define YYSTACK_GAP_MAXIMUM (YYSIZEOF (union yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define YYSTACK_BYTES(N) \
     ((N) * (YYSIZEOF (yy_state_t) + YYSIZEOF (YYSTYPE)) \
      + YYSTACK_GAP_MAXIMUM)

# define YYCOPY_NEEDED 1

/* Relocate STACK from its old location to the new one.  The
   local variables YYSIZE and YYSTACKSIZE give the old and new number of
   elements in the stack, and YYPTR gives the new location of the
   stack.  Advance YYPTR to a properly aligned location for the next
   stack.  */
# define YYSTACK_RELOCATE(Stack_alloc, Stack)                           \
    do                                                                  \
      {                                                                 \
        YYPTRDIFF_T yynewbytes;                                         \
        YYCOPY (&yyptr->Stack_alloc, Stack, yysize);                    \
        Stack = &yyptr->Stack_alloc;                                    \
        yynewbytes = yystacksize * YYSIZEOF (*Stack) + YYSTACK_GAP_MAXIMUM; \
        yyptr += yynewbytes / YYSIZEOF (*yyptr);                        \
      }                                                                 \
    while (0)

#endif

#if defined YYCOPY_NEEDED && YYCOPY_NEEDED
/* Copy COUNT objects from SRC to DST.  The source and destination do
   not overlap.  */
# ifndef YYCOPY
#  if defined __GNUC__ && 1 < __GNUC__
#   define YYCOPY(Dst, Src, Count) \
      __builtin_memcpy (Dst, Src, YY_CAST (YYSIZE_T, (Count)) * sizeof (*(Src)))
#  else
#   define YYCOPY(Dst, Src, Count)              \
      do                                        \
        {                                       \
          YYPTRDIFF_T yyi;                      \
          for (yyi = 0; yyi < (Count); yyi++)   \
            (Dst)[yyi] = (Src)[yyi];            \
        }                                       \
      while (0)
#  endif
# endif
#endif /* !YYCOPY_NEEDED */

/* YYFINAL -- State number of the termination state.  */
#define YYFINAL  5
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   288

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  85
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  32
/* YYNRULES -- Number of rules.  */
#define YYNRULES  81
/* YYNSTATES -- Number of states.  */
#define YYNSTATES  224

/* YYMAXUTOK -- Last valid token kind.  */
#define YYMAXUTOK   339


/* YYTRANSLATE(TOKEN-NUM) -- Symbol number corresponding to TOKEN-NUM
   as returned by yylex, with out-of-bounds checking.  */
#define YYTRANSLATE(YYX)                                \
  (0 <= (YYX) && (YYX) <= YYMAXUTOK                     \
   ? YY_CAST (yysymbol_kind_t, yytranslate[YYX])        \
   : YYSYMBOL_YYUNDEF)

/* YYTRANSLATE[TOKEN-NUM] -- Symbol number corresponding to TOKEN-NUM
   as returned by yylex.  */
static const yytype_int8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     1,     2,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    37,    38,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    50,    51,    52,    53,    54,
      55,    56,    57,    58,    59,    60,    61,    62,    63,    64,
      65,    66,    67,    68,    69,    70,    71,    72,    73,    74,
      75,    76,    77,    78,    79,    80,    81,    82,    83,    84
};

#if YYDEBUG
/* YYRLINE[YYN] -- Source line where rule number YYN was defined.  */
static const yytype_int16 yyrline[] =
{
       0,    62,    62,    64,    67,    68,    69,    71,    72,    75,
     107,   110,   116,   120,   121,   124,   128,   130,   132,   133,
     135,   137,   139,   143,   144,   146,   152,   154,   157,   163,
     168,   173,   175,   177,   179,   181,   183,   185,   188,   189,
     192,   195,   200,   202,   204,   206,   208,   212,   219,   222,
     230,   235,   240,   245,   250,   255,   260,   266,   269,   272,
     272,   272,   272,   274,   275,   278,   280,   282,   284,   287,
     288,   290,   290,   292,   292,   293,   293,   294,   294,   296,
     296,   297
};
#endif

/** Accessing symbol of state STATE.  */
#define YY_ACCESSING_SYMBOL(State) YY_CAST (yysymbol_kind_t, yystos[State])

#if YYDEBUG || 0
/* The user-facing name of the symbol whose (internal) number is
   YYSYMBOL.  No bounds checking.  */
static const char *yysymbol_name (yysymbol_kind_t yysymbol) YY_ATTRIBUTE_UNUSED;

/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "\"end of file\"", "error", "\"invalid token\"", "IDENTIFIER", "ASSIGN",
  "LOAD", "LPAREN", "RPAREN", "SEMICOLON", "QUOTE", "SET_UNWEIGHTED",
  "SET_UNDIRECTED", "MODEL_W", "EVAL", "TRAIN", "LAYER", "ITERS",
  "VAL_STEP", "AGGR_INIT", "FN_ARG", "MUL_SUM", "DSL_DOT", "FFN_OUT",
  "SIZE_FN", "GRAPH_ATTR", "FEAT_ATTR", "RELU", "LABEL_ATTR",
  "DEGREE_ATTR", "NODE_ATTR", "LEAKY_RELU", "POW", "SCALAR_INIT",
  "COLTILE", "FEAT_SIZE_ASSIGN", "LABEL_SIZE_ASSIGN", "COARSEN",
  "SRC_ATTR", "DST_ATTR", "INTEGER", "FLOAT", "SOFTMAX", "INIT_WEIGHT",
  "LBRACE", "RBRACE", "DOT", "COMMA", "TR", "FA", "PLUS", "MINUS",
  "MULTIPLY", "DIVIDE", "FFN", "NULL_KEY", "LOSS", "OPTIMIZER",
  "RMSE_LOSS", "ADAM_T", "DSL_FN", "RELAXNLN", "QUANT",
  "RABBIT_REORDER_OP", "SAMPLE_RANDOM_OP", "AGGR", "LSQBRA", "RSQBRA",
  "IF", "ELSE", "DO", "WHILE", "NOT", "AND", "OR", "NOTEQ", "EQ",
  "GREATER", "LESS", "GREATEREQ", "LESSEQ", "DATASET", "NONLN",
  "SENSEI_OP", "INT", "NEW", "$accept", "program", "load_dataset",
  "algorithm", "statements", "statement", "layers", "layer_def", "model",
  "model_def", "layer_inits", "layer_init", "model_init", "model_uses",
  "model_use", "gnn_op", "function", "update_op", "schedules", "schedule",
  "data_transform", "function_transform", "data_var", "function_init",
  "semiring_op", "op", "train_args", "train_arg", "args", "arg", "bool",
  "string", YY_NULLPTR
};

static const char *
yysymbol_name (yysymbol_kind_t yysymbol)
{
  return yytname[yysymbol];
}
#endif

#define YYPACT_NINF (-101)

#define yypact_value_is_default(Yyn) \
  ((Yyn) == YYPACT_NINF)

#define YYTABLE_NINF (-1)

#define yytable_value_is_error(Yyn) \
  0

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
static const yytype_int16 yypact[] =
{
      24,    12,    37,    38,    43,  -101,    45,    60,    38,    50,
    -101,    61,   111,  -101,    77,    81,    60,  -101,  -101,  -101,
      13,  -101,    69,  -101,  -101,   103,   106,    19,   115,   116,
      93,   104,    98,    47,   119,    89,    91,  -101,   128,   114,
      31,   129,   133,   141,   138,   128,   130,  -101,   134,   149,
     150,   151,   152,   153,   154,   156,  -101,  -101,  -101,  -101,
    -101,  -101,   128,   128,  -101,   157,   158,   118,   160,  -101,
    -101,  -101,   161,  -101,  -101,  -101,   162,   166,    15,  -101,
     133,   163,   165,     6,    32,     0,   167,   128,   128,   128,
     131,   128,   168,   128,   132,   132,   170,   171,    80,   169,
     173,  -101,   175,   179,   172,  -101,  -101,  -101,  -101,   128,
    -101,   146,   140,   159,   142,   143,   -23,  -101,   174,    21,
      47,    30,   177,    23,  -101,    55,  -101,  -101,   181,   184,
     185,   187,  -101,  -101,     3,  -101,   155,   188,    26,   190,
    -101,   164,  -101,  -101,  -101,   178,  -101,   128,   176,  -101,
    -101,   180,    63,    63,   182,   183,   186,     5,   191,  -101,
    -101,   192,     1,  -101,  -101,   194,    27,   196,   128,  -101,
    -101,   198,   199,   200,   201,  -101,   189,   203,    62,   204,
     207,  -101,  -101,  -101,  -101,  -101,    29,   205,   206,   209,
     210,    17,  -101,  -101,   211,   216,   219,  -101,  -101,    67,
    -101,  -101,  -101,  -101,  -101,   220,  -101,  -101,     8,  -101,
     193,   195,   212,   217,   197,   202,   221,  -101,  -101,  -101,
    -101,    11,   218,  -101
};

/* YYDEFACT[STATE-NUM] -- Default reduction number in state STATE-NUM.
   Performed when YYTABLE does not specify something else to do.  Zero
   means the default is an error.  */
static const yytype_int8 yydefact[] =
{
       0,     0,     0,     4,     0,     1,     0,     0,     4,     0,
      13,     0,     0,    48,     0,     0,     2,    38,    40,    41,
       0,     5,     0,    14,     6,     0,     0,    48,     0,     0,
       0,     0,    26,     0,     0,     0,     0,    39,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    69,     0,     0,
       0,     0,     0,     0,     0,     0,    30,     9,    59,    60,
      61,    62,     0,     0,    12,     0,     0,     0,     0,    51,
      50,    52,     0,    49,    55,    56,     0,     0,     0,    16,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    27,    25,     0,     0,     0,     0,
       0,    69,     0,     0,     0,    22,    81,     3,    29,     0,
      10,     0,     0,     0,    72,    74,    76,    70,     0,     0,
       0,     0,     0,     0,    37,     0,    44,    45,     0,     0,
       0,     0,    53,    54,     0,    69,     0,     0,     0,     0,
       7,    77,    71,    73,    75,     0,    32,     0,     0,    35,
      36,     0,     0,     0,     0,     0,     0,     0,     0,    63,
      28,     0,     0,    78,    58,     0,     0,     0,     0,    79,
      80,     0,     0,     0,     0,    18,     0,     0,     0,     0,
       0,    15,     8,    57,    33,    34,     0,     0,     0,     0,
       0,     0,    21,    69,     0,     0,     0,    64,    11,     0,
      31,    43,    42,    46,    47,     0,    17,    19,     0,    24,
       0,     0,     0,     0,    65,    67,     0,    23,    66,    68,
      69,     0,     0,    20
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -101,  -101,  -101,   222,  -101,    66,  -101,   224,  -101,  -101,
    -101,  -101,  -101,  -101,   208,  -101,  -101,  -101,  -101,   215,
    -101,  -101,    -7,  -101,  -101,   -31,  -101,  -101,  -100,  -101,
      82,  -101
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_uint8 yydefgoto[] =
{
       0,     2,     3,     7,   162,     8,     9,    10,    24,    25,
     191,   207,    42,    79,    80,    31,    32,    56,    16,    17,
      18,    19,   116,    34,   165,    62,   178,   197,    85,   117,
     171,    44
};

/* YYTABLE[YYPACT[STATE-NUM]] -- What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule whose
   number is the opposite.  If YYTABLE_NINF, syntax error.  */
static const yytype_uint8 yytable[] =
{
      20,   134,    63,    13,   180,    33,    13,   112,    13,    20,
     156,    13,   176,   108,    13,   213,     4,    38,   222,   103,
     205,   113,    39,   144,   113,    45,   113,     1,   146,   113,
     150,    67,   113,   160,   184,   157,   200,     5,    83,   114,
     110,     6,   114,    76,   114,   181,    28,   114,    11,    12,
     114,    39,   109,    22,   115,    94,    95,   115,    39,   115,
     104,   206,   115,    13,    46,   115,    39,    26,    39,   194,
      27,    39,    39,    40,    39,    39,   148,   111,   195,   196,
     119,   120,   121,    35,   123,    29,   125,    36,    30,   147,
     128,   129,    39,   208,    14,    15,    58,    59,    60,    61,
      39,   151,   138,    68,    69,    70,    41,    71,    72,    73,
     169,   170,    57,   130,    27,    43,   131,    74,    75,    49,
     221,    47,    48,    50,    51,    52,    28,    64,    65,    29,
      66,    13,    30,    77,    53,    54,    78,    68,    69,    70,
     166,    71,    72,    73,    81,    82,    55,    58,    59,    60,
      61,    74,    75,    86,    84,    87,    88,    89,    90,    91,
      92,   186,    93,    98,    96,    97,    99,   100,   101,   102,
     122,   118,   106,   107,   139,   124,   132,    39,   126,   127,
     133,   135,   136,   140,   149,   141,   137,   152,   142,   143,
     153,   154,    33,   155,   159,   145,   161,   192,   164,   179,
     158,   183,   168,   185,   177,   187,   188,   189,   190,   193,
     163,   199,   198,   201,   202,   216,   167,   203,   204,   209,
     210,   173,   174,   211,   212,   217,   223,   220,   182,   175,
      21,    37,   214,    23,   215,   172,     0,     0,     0,     0,
       0,     0,     0,   218,     0,     0,     0,     0,   219,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   105
};

static const yytype_int16 yycheck[] =
{
       7,   101,    33,     3,     3,    12,     3,     7,     3,    16,
       7,     3,     7,     7,     3,     7,     4,     4,     7,     4,
       3,    21,    45,    46,    21,     6,    21,     3,     7,    21,
       7,    38,    21,     7,     7,   135,     7,     0,    45,    39,
       8,     3,    39,    12,    39,    44,    15,    39,     5,     4,
      39,    45,    46,     3,    54,    62,    63,    54,    45,    54,
      45,    44,    54,     3,    45,    54,    45,     6,    45,     7,
       3,    45,    45,     4,    45,    45,    46,    45,    16,    17,
      87,    88,    89,     6,    91,    18,    93,     6,    21,   120,
      10,    11,    45,   193,    34,    35,    49,    50,    51,    52,
      45,    46,   109,    23,    24,    25,     3,    27,    28,    29,
      47,    48,     8,    33,     3,     9,    36,    37,    38,    26,
     220,     6,     6,    30,    31,    32,    15,     8,    39,    18,
      39,     3,    21,     4,    41,    42,     3,    23,    24,    25,
     147,    27,    28,    29,     3,     7,    53,    49,    50,    51,
      52,    37,    38,    19,    24,     6,     6,     6,     6,     6,
       6,   168,     6,    45,     7,     7,     6,     6,     6,     3,
      39,     4,     9,     8,    28,     7,     7,    45,     8,     8,
       7,     6,     3,    43,     7,    26,    14,     6,    46,    46,
       6,     6,   199,     6,     6,    21,     6,     8,    20,     7,
      45,     7,    22,     7,    13,     7,     7,     7,     7,     6,
      46,     4,     8,     8,     8,     3,    40,     8,     8,     8,
       4,    39,    39,     4,     4,     8,     8,     6,   162,    43,
       8,    16,    39,     9,    39,   153,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    46,    -1,    -1,    -1,    -1,    46,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    80
};

/* YYSTOS[STATE-NUM] -- The symbol kind of the accessing symbol of
   state STATE-NUM.  */
static const yytype_int8 yystos[] =
{
       0,     3,    86,    87,     4,     0,     3,    88,    90,    91,
      92,     5,     4,     3,    34,    35,   103,   104,   105,   106,
     107,    88,     3,    92,    93,    94,     6,     3,    15,    18,
      21,   100,   101,   107,   108,     6,     6,   104,     4,    45,
       4,     3,    97,     9,   116,     6,    45,     6,     6,    26,
      30,    31,    32,    41,    42,    53,   102,     8,    49,    50,
      51,    52,   110,   110,     8,    39,    39,   107,    23,    24,
      25,    27,    28,    29,    37,    38,    12,     4,     3,    98,
      99,     3,     7,   107,    24,   113,    19,     6,     6,     6,
       6,     6,     6,     6,   107,   107,     7,     7,    45,     6,
       6,     6,     3,     4,    45,    99,     9,     8,     7,    46,
       8,    45,     7,    21,    39,    54,   107,   114,     4,   107,
     107,   107,    39,   107,     7,   107,     8,     8,    10,    11,
      33,    36,     7,     7,   113,     6,     3,    14,   107,    28,
      43,    26,    46,    46,    46,    21,     7,   110,    46,     7,
       7,    46,     6,     6,     6,     6,     7,   113,    45,     6,
       7,     6,    89,    46,    20,   109,   107,    40,    22,    47,
      48,   115,   115,    39,    39,    43,     7,    13,   111,     7,
       3,    44,    90,     7,     7,     7,   107,     7,     7,     7,
       7,    95,     8,     6,     7,    16,    17,   112,     8,     4,
       7,     8,     8,     8,     8,     3,    44,    96,   113,     8,
       4,     4,     4,     7,    39,    39,     3,     8,    46,    46,
       6,   113,     7,     8
};

/* YYR1[RULE-NUM] -- Symbol kind of the left-hand side of rule RULE-NUM.  */
static const yytype_int8 yyr1[] =
{
       0,    85,    86,    87,    88,    88,    88,    89,    89,    90,
      90,    90,    90,    91,    91,    92,    93,    94,    95,    95,
      96,    97,    98,    99,    99,   100,   100,   100,   101,   101,
     101,   102,   102,   102,   102,   102,   102,   102,   103,   103,
     104,   104,   105,   105,   105,   105,   105,   106,   107,   107,
     107,   107,   107,   107,   107,   107,   107,   108,   109,   110,
     110,   110,   110,   111,   111,   112,   112,   112,   112,   113,
     113,   114,   114,   114,   114,   114,   114,   114,   114,   115,
     115,   116
};

/* YYR2[RULE-NUM] -- Number of symbols on the right-hand side of rule RULE-NUM.  */
static const yytype_int8 yyr2[] =
{
       0,     2,     3,     7,     0,     2,     2,     0,     2,     4,
       6,    10,     4,     1,     2,     9,     3,     9,     0,     2,
       7,     7,     2,     9,     7,     3,     1,     3,     6,     4,
       2,     7,     4,     6,     6,     4,     4,     3,     1,     2,
       1,     1,     9,     9,     5,     5,     9,     9,     1,     3,
       3,     3,     3,     5,     5,     3,     3,     7,     1,     1,
       1,     1,     1,     0,     2,     3,     4,     3,     4,     0,
       2,     2,     1,     2,     1,     2,     1,     2,     3,     1,
       1,     3
};


enum { YYENOMEM = -2 };

#define yyerrok         (yyerrstatus = 0)
#define yyclearin       (yychar = YYEMPTY)

#define YYACCEPT        goto yyacceptlab
#define YYABORT         goto yyabortlab
#define YYERROR         goto yyerrorlab
#define YYNOMEM         goto yyexhaustedlab


#define YYRECOVERING()  (!!yyerrstatus)

#define YYBACKUP(Token, Value)                                    \
  do                                                              \
    if (yychar == YYEMPTY)                                        \
      {                                                           \
        yychar = (Token);                                         \
        yylval = (Value);                                         \
        YYPOPSTACK (yylen);                                       \
        yystate = *yyssp;                                         \
        goto yybackup;                                            \
      }                                                           \
    else                                                          \
      {                                                           \
        yyerror (YY_("syntax error: cannot back up")); \
        YYERROR;                                                  \
      }                                                           \
  while (0)

/* Backward compatibility with an undocumented macro.
   Use YYerror or YYUNDEF. */
#define YYERRCODE YYUNDEF


/* Enable debugging if requested.  */
#if YYDEBUG

# ifndef YYFPRINTF
#  include <stdio.h> /* INFRINGES ON USER NAME SPACE */
#  define YYFPRINTF fprintf
# endif

# define YYDPRINTF(Args)                        \
do {                                            \
  if (yydebug)                                  \
    YYFPRINTF Args;                             \
} while (0)




# define YY_SYMBOL_PRINT(Title, Kind, Value, Location)                    \
do {                                                                      \
  if (yydebug)                                                            \
    {                                                                     \
      YYFPRINTF (stderr, "%s ", Title);                                   \
      yy_symbol_print (stderr,                                            \
                  Kind, Value); \
      YYFPRINTF (stderr, "\n");                                           \
    }                                                                     \
} while (0)


/*-----------------------------------.
| Print this symbol's value on YYO.  |
`-----------------------------------*/

static void
yy_symbol_value_print (FILE *yyo,
                       yysymbol_kind_t yykind, YYSTYPE const * const yyvaluep)
{
  FILE *yyoutput = yyo;
  YY_USE (yyoutput);
  if (!yyvaluep)
    return;
  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  YY_USE (yykind);
  YY_IGNORE_MAYBE_UNINITIALIZED_END
}


/*---------------------------.
| Print this symbol on YYO.  |
`---------------------------*/

static void
yy_symbol_print (FILE *yyo,
                 yysymbol_kind_t yykind, YYSTYPE const * const yyvaluep)
{
  YYFPRINTF (yyo, "%s %s (",
             yykind < YYNTOKENS ? "token" : "nterm", yysymbol_name (yykind));

  yy_symbol_value_print (yyo, yykind, yyvaluep);
  YYFPRINTF (yyo, ")");
}

/*------------------------------------------------------------------.
| yy_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (included).                                                   |
`------------------------------------------------------------------*/

static void
yy_stack_print (yy_state_t *yybottom, yy_state_t *yytop)
{
  YYFPRINTF (stderr, "Stack now");
  for (; yybottom <= yytop; yybottom++)
    {
      int yybot = *yybottom;
      YYFPRINTF (stderr, " %d", yybot);
    }
  YYFPRINTF (stderr, "\n");
}

# define YY_STACK_PRINT(Bottom, Top)                            \
do {                                                            \
  if (yydebug)                                                  \
    yy_stack_print ((Bottom), (Top));                           \
} while (0)


/*------------------------------------------------.
| Report that the YYRULE is going to be reduced.  |
`------------------------------------------------*/

static void
yy_reduce_print (yy_state_t *yyssp, YYSTYPE *yyvsp,
                 int yyrule)
{
  int yylno = yyrline[yyrule];
  int yynrhs = yyr2[yyrule];
  int yyi;
  YYFPRINTF (stderr, "Reducing stack by rule %d (line %d):\n",
             yyrule - 1, yylno);
  /* The symbols being reduced.  */
  for (yyi = 0; yyi < yynrhs; yyi++)
    {
      YYFPRINTF (stderr, "   $%d = ", yyi + 1);
      yy_symbol_print (stderr,
                       YY_ACCESSING_SYMBOL (+yyssp[yyi + 1 - yynrhs]),
                       &yyvsp[(yyi + 1) - (yynrhs)]);
      YYFPRINTF (stderr, "\n");
    }
}

# define YY_REDUCE_PRINT(Rule)          \
do {                                    \
  if (yydebug)                          \
    yy_reduce_print (yyssp, yyvsp, Rule); \
} while (0)

/* Nonzero means print parse trace.  It is left uninitialized so that
   multiple parsers can coexist.  */
int yydebug;
#else /* !YYDEBUG */
# define YYDPRINTF(Args) ((void) 0)
# define YY_SYMBOL_PRINT(Title, Kind, Value, Location)
# define YY_STACK_PRINT(Bottom, Top)
# define YY_REDUCE_PRINT(Rule)
#endif /* !YYDEBUG */


/* YYINITDEPTH -- initial size of the parser's stacks.  */
#ifndef YYINITDEPTH
# define YYINITDEPTH 200
#endif

/* YYMAXDEPTH -- maximum size the stacks can grow to (effective only
   if the built-in stack extension method is used).

   Do not make this value too large; the results are undefined if
   YYSTACK_ALLOC_MAXIMUM < YYSTACK_BYTES (YYMAXDEPTH)
   evaluated with infinite-precision integer arithmetic.  */

#ifndef YYMAXDEPTH
# define YYMAXDEPTH 10000
#endif






/*-----------------------------------------------.
| Release the memory associated to this symbol.  |
`-----------------------------------------------*/

static void
yydestruct (const char *yymsg,
            yysymbol_kind_t yykind, YYSTYPE *yyvaluep)
{
  YY_USE (yyvaluep);
  if (!yymsg)
    yymsg = "Deleting";
  YY_SYMBOL_PRINT (yymsg, yykind, yyvaluep, yylocationp);

  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  YY_USE (yykind);
  YY_IGNORE_MAYBE_UNINITIALIZED_END
}


/* Lookahead token kind.  */
int yychar;

/* The semantic value of the lookahead symbol.  */
YYSTYPE yylval;
/* Number of syntax errors so far.  */
int yynerrs;




/*----------.
| yyparse.  |
`----------*/

int
yyparse (void)
{
    yy_state_fast_t yystate = 0;
    /* Number of tokens to shift before error messages enabled.  */
    int yyerrstatus = 0;

    /* Refer to the stacks through separate pointers, to allow yyoverflow
       to reallocate them elsewhere.  */

    /* Their size.  */
    YYPTRDIFF_T yystacksize = YYINITDEPTH;

    /* The state stack: array, bottom, top.  */
    yy_state_t yyssa[YYINITDEPTH];
    yy_state_t *yyss = yyssa;
    yy_state_t *yyssp = yyss;

    /* The semantic value stack: array, bottom, top.  */
    YYSTYPE yyvsa[YYINITDEPTH];
    YYSTYPE *yyvs = yyvsa;
    YYSTYPE *yyvsp = yyvs;

  int yyn;
  /* The return value of yyparse.  */
  int yyresult;
  /* Lookahead symbol kind.  */
  yysymbol_kind_t yytoken = YYSYMBOL_YYEMPTY;
  /* The variables used to return semantic value and location from the
     action routines.  */
  YYSTYPE yyval;



#define YYPOPSTACK(N)   (yyvsp -= (N), yyssp -= (N))

  /* The number of symbols on the RHS of the reduced rule.
     Keep to zero when no symbol should be popped.  */
  int yylen = 0;

  YYDPRINTF ((stderr, "Starting parse\n"));

  yychar = YYEMPTY; /* Cause a token to be read.  */

  goto yysetstate;


/*------------------------------------------------------------.
| yynewstate -- push a new state, which is found in yystate.  |
`------------------------------------------------------------*/
yynewstate:
  /* In all cases, when you get here, the value and location stacks
     have just been pushed.  So pushing a state here evens the stacks.  */
  yyssp++;


/*--------------------------------------------------------------------.
| yysetstate -- set current state (the top of the stack) to yystate.  |
`--------------------------------------------------------------------*/
yysetstate:
  YYDPRINTF ((stderr, "Entering state %d\n", yystate));
  YY_ASSERT (0 <= yystate && yystate < YYNSTATES);
  YY_IGNORE_USELESS_CAST_BEGIN
  *yyssp = YY_CAST (yy_state_t, yystate);
  YY_IGNORE_USELESS_CAST_END
  YY_STACK_PRINT (yyss, yyssp);

  if (yyss + yystacksize - 1 <= yyssp)
#if !defined yyoverflow && !defined YYSTACK_RELOCATE
    YYNOMEM;
#else
    {
      /* Get the current used size of the three stacks, in elements.  */
      YYPTRDIFF_T yysize = yyssp - yyss + 1;

# if defined yyoverflow
      {
        /* Give user a chance to reallocate the stack.  Use copies of
           these so that the &'s don't force the real ones into
           memory.  */
        yy_state_t *yyss1 = yyss;
        YYSTYPE *yyvs1 = yyvs;

        /* Each stack pointer address is followed by the size of the
           data in use in that stack, in bytes.  This used to be a
           conditional around just the two extra args, but that might
           be undefined if yyoverflow is a macro.  */
        yyoverflow (YY_("memory exhausted"),
                    &yyss1, yysize * YYSIZEOF (*yyssp),
                    &yyvs1, yysize * YYSIZEOF (*yyvsp),
                    &yystacksize);
        yyss = yyss1;
        yyvs = yyvs1;
      }
# else /* defined YYSTACK_RELOCATE */
      /* Extend the stack our own way.  */
      if (YYMAXDEPTH <= yystacksize)
        YYNOMEM;
      yystacksize *= 2;
      if (YYMAXDEPTH < yystacksize)
        yystacksize = YYMAXDEPTH;

      {
        yy_state_t *yyss1 = yyss;
        union yyalloc *yyptr =
          YY_CAST (union yyalloc *,
                   YYSTACK_ALLOC (YY_CAST (YYSIZE_T, YYSTACK_BYTES (yystacksize))));
        if (! yyptr)
          YYNOMEM;
        YYSTACK_RELOCATE (yyss_alloc, yyss);
        YYSTACK_RELOCATE (yyvs_alloc, yyvs);
#  undef YYSTACK_RELOCATE
        if (yyss1 != yyssa)
          YYSTACK_FREE (yyss1);
      }
# endif

      yyssp = yyss + yysize - 1;
      yyvsp = yyvs + yysize - 1;

      YY_IGNORE_USELESS_CAST_BEGIN
      YYDPRINTF ((stderr, "Stack size increased to %ld\n",
                  YY_CAST (long, yystacksize)));
      YY_IGNORE_USELESS_CAST_END

      if (yyss + yystacksize - 1 <= yyssp)
        YYABORT;
    }
#endif /* !defined yyoverflow && !defined YYSTACK_RELOCATE */


  if (yystate == YYFINAL)
    YYACCEPT;

  goto yybackup;


/*-----------.
| yybackup.  |
`-----------*/
yybackup:
  /* Do appropriate processing given the current state.  Read a
     lookahead token if we need one and don't already have one.  */

  /* First try to decide what to do without reference to lookahead token.  */
  yyn = yypact[yystate];
  if (yypact_value_is_default (yyn))
    goto yydefault;

  /* Not known => get a lookahead token if don't already have one.  */

  /* YYCHAR is either empty, or end-of-input, or a valid lookahead.  */
  if (yychar == YYEMPTY)
    {
      YYDPRINTF ((stderr, "Reading a token\n"));
      yychar = yylex ();
    }

  if (yychar <= YYEOF)
    {
      yychar = YYEOF;
      yytoken = YYSYMBOL_YYEOF;
      YYDPRINTF ((stderr, "Now at end of input.\n"));
    }
  else if (yychar == YYerror)
    {
      /* The scanner already issued an error message, process directly
         to error recovery.  But do not keep the error token as
         lookahead, it is too special and may lead us to an endless
         loop in error recovery. */
      yychar = YYUNDEF;
      yytoken = YYSYMBOL_YYerror;
      goto yyerrlab1;
    }
  else
    {
      yytoken = YYTRANSLATE (yychar);
      YY_SYMBOL_PRINT ("Next token is", yytoken, &yylval, &yylloc);
    }

  /* If the proper action on seeing token YYTOKEN is to reduce or to
     detect an error, take that action.  */
  yyn += yytoken;
  if (yyn < 0 || YYLAST < yyn || yycheck[yyn] != yytoken)
    goto yydefault;
  yyn = yytable[yyn];
  if (yyn <= 0)
    {
      if (yytable_value_is_error (yyn))
        goto yyerrlab;
      yyn = -yyn;
      goto yyreduce;
    }

  /* Count tokens shifted since error; after three, turn off error
     status.  */
  if (yyerrstatus)
    yyerrstatus--;

  /* Shift the lookahead token.  */
  YY_SYMBOL_PRINT ("Shifting", yytoken, &yylval, &yylloc);
  yystate = yyn;
  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END

  /* Discard the shifted token.  */
  yychar = YYEMPTY;
  goto yynewstate;


/*-----------------------------------------------------------.
| yydefault -- do the default action for the current state.  |
`-----------------------------------------------------------*/
yydefault:
  yyn = yydefact[yystate];
  if (yyn == 0)
    goto yyerrlab;
  goto yyreduce;


/*-----------------------------.
| yyreduce -- do a reduction.  |
`-----------------------------*/
yyreduce:
  /* yyn is the number of a rule to reduce with.  */
  yylen = yyr2[yyn];

  /* If YYLEN is nonzero, implement the default value of the action:
     '$$ = $1'.

     Otherwise, the following line sets YYVAL to garbage.
     This behavior is undocumented and Bison
     users should not rely upon it.  Assigning to YYVAL
     unconditionally makes the parser a bit smaller, and it avoids a
     GCC warning that YYVAL may be used uninitialized.  */
  yyval = yyvsp[1-yylen];


  YY_REDUCE_PRINT (yyn);
  switch (yyn)
    {
  case 2: /* program: load_dataset algorithm schedules  */
#line 62 "frontend.y"
                                           {}
#line 1367 "build/frontend.tab.c"
    break;

  case 3: /* load_dataset: IDENTIFIER ASSIGN LOAD LPAREN string RPAREN SEMICOLON  */
#line 65 "frontend.y"
    { m1.dataset_name = (yyvsp[-2].sval); }
#line 1373 "build/frontend.tab.c"
    break;

  case 4: /* algorithm: %empty  */
#line 67 "frontend.y"
            { }
#line 1379 "build/frontend.tab.c"
    break;

  case 5: /* algorithm: statement algorithm  */
#line 68 "frontend.y"
                          {}
#line 1385 "build/frontend.tab.c"
    break;

  case 6: /* algorithm: layers model  */
#line 69 "frontend.y"
                   {}
#line 1391 "build/frontend.tab.c"
    break;

  case 7: /* statements: %empty  */
#line 71 "frontend.y"
             { }
#line 1397 "build/frontend.tab.c"
    break;

  case 8: /* statements: statements statement  */
#line 73 "frontend.y"
    {}
#line 1403 "build/frontend.tab.c"
    break;

  case 9: /* statement: IDENTIFIER ASSIGN gnn_op SEMICOLON  */
#line 76 "frontend.y"
    {
        if ((yyvsp[-1].ltype) == GET_DEGREES){
            if (debug == 2) cout << "layer op - get degrees\n";
            m1.layer_operations.push_back((yyvsp[-1].ltype));
        }
        else if ((yyvsp[-1].ltype) == GET_NORMALIZATION){
            if (debug == 2) cout << "layer op - get normalization\n";
            m1.layer_operations.push_back((yyvsp[-1].ltype));

        }
        else if ((yyvsp[-1].ltype) == MULT_NORM_RES){
            if (debug == 2) cout << "layer op - mult norm res\n";
            m1.layer_operations.push_back((yyvsp[-1].ltype));

        }
        else if ((yyvsp[-1].ltype) == MESSAGE_PASSING_AGGREGATE){
            if (debug == 2) cout << "layer op - aggregate\n";
            m1.layer_operations.push_back((yyvsp[-1].ltype));

        }
        else if ((yyvsp[-1].ltype) == FEED_FORWARD_NN){
            if (debug == 2) cout << "layer op - ffn\n";
            m1.layer_operations.push_back((yyvsp[-1].ltype));

        }
        else if ((yyvsp[-1].ltype) == NON_LINEARITY){
            if (debug == 2) cout << "layer op - nonln\n";
            m1.layer_operations.push_back((yyvsp[-1].ltype));

        }
    }
#line 1439 "build/frontend.tab.c"
    break;

  case 10: /* statement: IDENTIFIER ASSIGN IDENTIFIER DOT GRAPH_ATTR SEMICOLON  */
#line 108 "frontend.y"
    {
    }
#line 1446 "build/frontend.tab.c"
    break;

  case 11: /* statement: IDENTIFIER ASSIGN IDENTIFIER DOT GRAPH_ATTR DOT DEGREE_ATTR LPAREN RPAREN SEMICOLON  */
#line 111 "frontend.y"
    {
        if (debug == 2) cout << "layer op - get degrees\n";
        m1.layer_operations.push_back(GET_DEGREES);

    }
#line 1456 "build/frontend.tab.c"
    break;

  case 12: /* statement: IDENTIFIER ASSIGN function_init SEMICOLON  */
#line 117 "frontend.y"
    {
    }
#line 1463 "build/frontend.tab.c"
    break;

  case 13: /* layers: layer_def  */
#line 120 "frontend.y"
                   {}
#line 1469 "build/frontend.tab.c"
    break;

  case 14: /* layers: layers layer_def  */
#line 122 "frontend.y"
    {}
#line 1475 "build/frontend.tab.c"
    break;

  case 15: /* layer_def: IDENTIFIER ASSIGN LAYER LPAREN args RPAREN LBRACE statements RBRACE  */
#line 125 "frontend.y"
    {}
#line 1481 "build/frontend.tab.c"
    break;

  case 16: /* model: model_def model_init model_uses  */
#line 128 "frontend.y"
                                        {}
#line 1487 "build/frontend.tab.c"
    break;

  case 17: /* model_def: IDENTIFIER ASSIGN MODEL_W LPAREN args RPAREN LBRACE layer_inits RBRACE  */
#line 130 "frontend.y"
                                                                                   {}
#line 1493 "build/frontend.tab.c"
    break;

  case 18: /* layer_inits: %empty  */
#line 132 "frontend.y"
              { }
#line 1499 "build/frontend.tab.c"
    break;

  case 19: /* layer_inits: layer_inits layer_init  */
#line 133 "frontend.y"
                             {}
#line 1505 "build/frontend.tab.c"
    break;

  case 20: /* layer_init: IDENTIFIER ASSIGN IDENTIFIER LPAREN args RPAREN SEMICOLON  */
#line 135 "frontend.y"
                                                                       { m1.num_layers++; }
#line 1511 "build/frontend.tab.c"
    break;

  case 21: /* model_init: IDENTIFIER ASSIGN IDENTIFIER LPAREN args RPAREN SEMICOLON  */
#line 137 "frontend.y"
                                                                       { }
#line 1517 "build/frontend.tab.c"
    break;

  case 22: /* model_uses: model_use model_use  */
#line 139 "frontend.y"
                                 {  }
#line 1523 "build/frontend.tab.c"
    break;

  case 23: /* model_use: IDENTIFIER ASSIGN IDENTIFIER DOT EVAL LPAREN args RPAREN SEMICOLON  */
#line 143 "frontend.y"
                                                                               {  }
#line 1529 "build/frontend.tab.c"
    break;

  case 24: /* model_use: IDENTIFIER DOT TRAIN LPAREN train_args RPAREN SEMICOLON  */
#line 144 "frontend.y"
                                                              {  }
#line 1535 "build/frontend.tab.c"
    break;

  case 25: /* gnn_op: data_var op data_var  */
#line 147 "frontend.y"
    {
        if ((yyvsp[-1].ival) == 3){
            (yyval.ltype) = MULT_NORM_RES;
        }
    }
#line 1545 "build/frontend.tab.c"
    break;

  case 26: /* gnn_op: function  */
#line 153 "frontend.y"
    { (yyval.ltype) = (yyvsp[0].ltype); }
#line 1551 "build/frontend.tab.c"
    break;

  case 27: /* gnn_op: function op data_var  */
#line 155 "frontend.y"
    {}
#line 1557 "build/frontend.tab.c"
    break;

  case 28: /* function: IDENTIFIER LPAREN data_var COMMA data_var RPAREN  */
#line 158 "frontend.y"
    {
        // for now we basically know this is an aggregate (but will need to update later)
        (yyval.ltype) = MESSAGE_PASSING_AGGREGATE;

    }
#line 1567 "build/frontend.tab.c"
    break;

  case 29: /* function: IDENTIFIER LPAREN data_var RPAREN  */
#line 164 "frontend.y"
    {
        // similarly know that this is relu
        (yyval.ltype) = NON_LINEARITY;
    }
#line 1576 "build/frontend.tab.c"
    break;

  case 30: /* function: DSL_DOT update_op  */
#line 169 "frontend.y"
    {
        (yyval.ltype) = (yyvsp[0].ltype);
    }
#line 1584 "build/frontend.tab.c"
    break;

  case 31: /* update_op: FFN LPAREN data_var COMMA FFN_OUT data_var RPAREN  */
#line 174 "frontend.y"
    { (yyval.ltype) = FEED_FORWARD_NN; }
#line 1590 "build/frontend.tab.c"
    break;

  case 32: /* update_op: RELU LPAREN data_var RPAREN  */
#line 176 "frontend.y"
    { (yyval.ltype) = NON_LINEARITY; }
#line 1596 "build/frontend.tab.c"
    break;

  case 33: /* update_op: LEAKY_RELU LPAREN data_var op data_var RPAREN  */
#line 178 "frontend.y"
    { }
#line 1602 "build/frontend.tab.c"
    break;

  case 34: /* update_op: POW LPAREN data_var COMMA FLOAT RPAREN  */
#line 180 "frontend.y"
    { (yyval.ltype) = GET_NORMALIZATION; }
#line 1608 "build/frontend.tab.c"
    break;

  case 35: /* update_op: SCALAR_INIT LPAREN INTEGER RPAREN  */
#line 182 "frontend.y"
    {}
#line 1614 "build/frontend.tab.c"
    break;

  case 36: /* update_op: SOFTMAX LPAREN data_var RPAREN  */
#line 184 "frontend.y"
    {}
#line 1620 "build/frontend.tab.c"
    break;

  case 37: /* update_op: INIT_WEIGHT LPAREN RPAREN  */
#line 186 "frontend.y"
    {}
#line 1626 "build/frontend.tab.c"
    break;

  case 38: /* schedules: schedule  */
#line 188 "frontend.y"
                     {}
#line 1632 "build/frontend.tab.c"
    break;

  case 39: /* schedules: schedules schedule  */
#line 190 "frontend.y"
    {}
#line 1638 "build/frontend.tab.c"
    break;

  case 40: /* schedule: data_transform  */
#line 193 "frontend.y"
    {
    }
#line 1645 "build/frontend.tab.c"
    break;

  case 41: /* schedule: function_transform  */
#line 196 "frontend.y"
    {
    }
#line 1652 "build/frontend.tab.c"
    break;

  case 42: /* data_transform: data_var ASSIGN data_var DOT SET_UNDIRECTED LPAREN bool RPAREN SEMICOLON  */
#line 201 "frontend.y"
    {  m1.addGraphTransformation(UNDIRECTED, (float) !(yyvsp[-2].ival));  }
#line 1658 "build/frontend.tab.c"
    break;

  case 43: /* data_transform: data_var ASSIGN data_var DOT SET_UNWEIGHTED LPAREN bool RPAREN SEMICOLON  */
#line 203 "frontend.y"
    {  m1.addGraphTransformation(UNWEIGHTED, (float) !(yyvsp[-2].ival)); }
#line 1664 "build/frontend.tab.c"
    break;

  case 44: /* data_transform: FEAT_SIZE_ASSIGN LPAREN INTEGER RPAREN SEMICOLON  */
#line 205 "frontend.y"
    {  m1.addGraphTransformation(FEAT_SIZE, atof((yyvsp[-2].sval)));  }
#line 1670 "build/frontend.tab.c"
    break;

  case 45: /* data_transform: LABEL_SIZE_ASSIGN LPAREN INTEGER RPAREN SEMICOLON  */
#line 207 "frontend.y"
    {  m1.addGraphTransformation(LABEL_SIZE, atof((yyvsp[-2].sval)));  }
#line 1676 "build/frontend.tab.c"
    break;

  case 46: /* data_transform: data_var ASSIGN data_var DOT COLTILE LPAREN INTEGER RPAREN SEMICOLON  */
#line 209 "frontend.y"
    {  m1.addDataTransformation(COL_TILE, atof((yyvsp[-2].sval)));  }
#line 1682 "build/frontend.tab.c"
    break;

  case 47: /* function_transform: data_var ASSIGN data_var DOT COARSEN LPAREN INTEGER RPAREN SEMICOLON  */
#line 213 "frontend.y"
    {
        m1.addComputeTransformation(COARSE, atof((yyvsp[-2].sval)));
    }
#line 1690 "build/frontend.tab.c"
    break;

  case 48: /* data_var: IDENTIFIER  */
#line 220 "frontend.y"
    {
    }
#line 1697 "build/frontend.tab.c"
    break;

  case 49: /* data_var: data_var DOT NODE_ATTR  */
#line 223 "frontend.y"
    {
        if ((yyvsp[-2].sval) == "feats"){
            (yyval.sval) = (yyvsp[-2].sval);
        }
        (yyval.sval) = strdup("node");
        free((yyvsp[-2].sval));
    }
#line 1709 "build/frontend.tab.c"
    break;

  case 50: /* data_var: data_var DOT FEAT_ATTR  */
#line 231 "frontend.y"
    {
        (yyval.sval) = strdup("feats");
        free((yyvsp[-2].sval));
    }
#line 1718 "build/frontend.tab.c"
    break;

  case 51: /* data_var: data_var DOT GRAPH_ATTR  */
#line 236 "frontend.y"
    {
        (yyval.sval) = strdup("graphs");
        free((yyvsp[-2].sval));
    }
#line 1727 "build/frontend.tab.c"
    break;

  case 52: /* data_var: data_var DOT LABEL_ATTR  */
#line 241 "frontend.y"
    {
        (yyval.sval) = strdup("label");
        free((yyvsp[-2].sval));
    }
#line 1736 "build/frontend.tab.c"
    break;

  case 53: /* data_var: data_var DOT SIZE_FN LPAREN RPAREN  */
#line 246 "frontend.y"
    {
        (yyval.sval) = strdup("size");
        free((yyvsp[-4].sval));
    }
#line 1745 "build/frontend.tab.c"
    break;

  case 54: /* data_var: data_var DOT DEGREE_ATTR LPAREN RPAREN  */
#line 251 "frontend.y"
    {
        (yyval.sval) = strdup("degrees");
        free((yyvsp[-4].sval));
    }
#line 1754 "build/frontend.tab.c"
    break;

  case 55: /* data_var: data_var DOT SRC_ATTR  */
#line 256 "frontend.y"
    {
        (yyval.sval) = strdup("src_nodes");
        free((yyvsp[-2].sval));
    }
#line 1763 "build/frontend.tab.c"
    break;

  case 56: /* data_var: data_var DOT DST_ATTR  */
#line 261 "frontend.y"
    {
        (yyval.sval) = strdup("dst_nodes");
        free((yyvsp[-2].sval));
    }
#line 1772 "build/frontend.tab.c"
    break;

  case 57: /* function_init: AGGR_INIT LPAREN FN_ARG ASSIGN DSL_DOT semiring_op RPAREN  */
#line 267 "frontend.y"
    {}
#line 1778 "build/frontend.tab.c"
    break;

  case 58: /* semiring_op: MUL_SUM  */
#line 270 "frontend.y"
    {}
#line 1784 "build/frontend.tab.c"
    break;

  case 59: /* op: PLUS  */
#line 272 "frontend.y"
          { (yyval.ival) = 1; }
#line 1790 "build/frontend.tab.c"
    break;

  case 60: /* op: MINUS  */
#line 272 "frontend.y"
                              { (yyval.ival) = 2; }
#line 1796 "build/frontend.tab.c"
    break;

  case 61: /* op: MULTIPLY  */
#line 272 "frontend.y"
                                                     { (yyval.ival) = 3; }
#line 1802 "build/frontend.tab.c"
    break;

  case 62: /* op: DIVIDE  */
#line 272 "frontend.y"
                                                                          { (yyval.ival) = 4;}
#line 1808 "build/frontend.tab.c"
    break;

  case 63: /* train_args: %empty  */
#line 274 "frontend.y"
             {  }
#line 1814 "build/frontend.tab.c"
    break;

  case 64: /* train_args: train_args train_arg  */
#line 276 "frontend.y"
    {}
#line 1820 "build/frontend.tab.c"
    break;

  case 65: /* train_arg: ITERS ASSIGN INTEGER  */
#line 279 "frontend.y"
    {  m1.iterations = atoi((yyvsp[0].sval)); free((yyvsp[0].sval)); }
#line 1826 "build/frontend.tab.c"
    break;

  case 66: /* train_arg: ITERS ASSIGN INTEGER COMMA  */
#line 281 "frontend.y"
    { m1.iterations = atoi((yyvsp[-1].sval)); free((yyvsp[-1].sval)); }
#line 1832 "build/frontend.tab.c"
    break;

  case 67: /* train_arg: VAL_STEP ASSIGN INTEGER  */
#line 283 "frontend.y"
    { m1.validation_step = atoi((yyvsp[0].sval)); free((yyvsp[0].sval)); }
#line 1838 "build/frontend.tab.c"
    break;

  case 68: /* train_arg: VAL_STEP ASSIGN INTEGER COMMA  */
#line 285 "frontend.y"
    { m1.validation_step = atoi((yyvsp[-1].sval)); free((yyvsp[-1].sval)); }
#line 1844 "build/frontend.tab.c"
    break;

  case 69: /* args: %empty  */
#line 287 "frontend.y"
       {  }
#line 1850 "build/frontend.tab.c"
    break;

  case 70: /* args: args arg  */
#line 288 "frontend.y"
               {}
#line 1856 "build/frontend.tab.c"
    break;

  case 71: /* arg: INTEGER COMMA  */
#line 290 "frontend.y"
                    { m1.output_input_classes = atof((yyvsp[-1].sval)); }
#line 1862 "build/frontend.tab.c"
    break;

  case 72: /* arg: INTEGER  */
#line 291 "frontend.y"
    { m1.output_input_classes = atof((yyvsp[0].sval)); }
#line 1868 "build/frontend.tab.c"
    break;

  case 74: /* arg: NULL_KEY  */
#line 292 "frontend.y"
                                {}
#line 1874 "build/frontend.tab.c"
    break;

  case 75: /* arg: data_var COMMA  */
#line 293 "frontend.y"
                     { }
#line 1880 "build/frontend.tab.c"
    break;

  case 76: /* arg: data_var  */
#line 293 "frontend.y"
                                    {}
#line 1886 "build/frontend.tab.c"
    break;

  case 78: /* arg: DSL_DOT RELU COMMA  */
#line 294 "frontend.y"
                                         {}
#line 1892 "build/frontend.tab.c"
    break;

  case 79: /* bool: TR  */
#line 296 "frontend.y"
          { (yyval.ival) = 1; }
#line 1898 "build/frontend.tab.c"
    break;

  case 80: /* bool: FA  */
#line 296 "frontend.y"
                           { (yyval.ival) = 0; }
#line 1904 "build/frontend.tab.c"
    break;

  case 81: /* string: QUOTE IDENTIFIER QUOTE  */
#line 298 "frontend.y"
    {
        (yyval.sval) = (char*) malloc(strlen((yyvsp[-1].sval)) + 2);
        sprintf((yyval.sval), "%s", (yyvsp[-1].sval));
        free((yyvsp[-1].sval));
    }
#line 1914 "build/frontend.tab.c"
    break;


#line 1918 "build/frontend.tab.c"

      default: break;
    }
  /* User semantic actions sometimes alter yychar, and that requires
     that yytoken be updated with the new translation.  We take the
     approach of translating immediately before every use of yytoken.
     One alternative is translating here after every semantic action,
     but that translation would be missed if the semantic action invokes
     YYABORT, YYACCEPT, or YYERROR immediately after altering yychar or
     if it invokes YYBACKUP.  In the case of YYABORT or YYACCEPT, an
     incorrect destructor might then be invoked immediately.  In the
     case of YYERROR or YYBACKUP, subsequent parser actions might lead
     to an incorrect destructor call or verbose syntax error message
     before the lookahead is translated.  */
  YY_SYMBOL_PRINT ("-> $$ =", YY_CAST (yysymbol_kind_t, yyr1[yyn]), &yyval, &yyloc);

  YYPOPSTACK (yylen);
  yylen = 0;

  *++yyvsp = yyval;

  /* Now 'shift' the result of the reduction.  Determine what state
     that goes to, based on the state we popped back to and the rule
     number reduced by.  */
  {
    const int yylhs = yyr1[yyn] - YYNTOKENS;
    const int yyi = yypgoto[yylhs] + *yyssp;
    yystate = (0 <= yyi && yyi <= YYLAST && yycheck[yyi] == *yyssp
               ? yytable[yyi]
               : yydefgoto[yylhs]);
  }

  goto yynewstate;


/*--------------------------------------.
| yyerrlab -- here on detecting error.  |
`--------------------------------------*/
yyerrlab:
  /* Make sure we have latest lookahead translation.  See comments at
     user semantic actions for why this is necessary.  */
  yytoken = yychar == YYEMPTY ? YYSYMBOL_YYEMPTY : YYTRANSLATE (yychar);
  /* If not already recovering from an error, report this error.  */
  if (!yyerrstatus)
    {
      ++yynerrs;
      yyerror (YY_("syntax error"));
    }

  if (yyerrstatus == 3)
    {
      /* If just tried and failed to reuse lookahead token after an
         error, discard it.  */

      if (yychar <= YYEOF)
        {
          /* Return failure if at end of input.  */
          if (yychar == YYEOF)
            YYABORT;
        }
      else
        {
          yydestruct ("Error: discarding",
                      yytoken, &yylval);
          yychar = YYEMPTY;
        }
    }

  /* Else will try to reuse lookahead token after shifting the error
     token.  */
  goto yyerrlab1;


/*---------------------------------------------------.
| yyerrorlab -- error raised explicitly by YYERROR.  |
`---------------------------------------------------*/
yyerrorlab:
  /* Pacify compilers when the user code never invokes YYERROR and the
     label yyerrorlab therefore never appears in user code.  */
  if (0)
    YYERROR;
  ++yynerrs;

  /* Do not reclaim the symbols of the rule whose action triggered
     this YYERROR.  */
  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);
  yystate = *yyssp;
  goto yyerrlab1;


/*-------------------------------------------------------------.
| yyerrlab1 -- common code for both syntax error and YYERROR.  |
`-------------------------------------------------------------*/
yyerrlab1:
  yyerrstatus = 3;      /* Each real token shifted decrements this.  */

  /* Pop stack until we find a state that shifts the error token.  */
  for (;;)
    {
      yyn = yypact[yystate];
      if (!yypact_value_is_default (yyn))
        {
          yyn += YYSYMBOL_YYerror;
          if (0 <= yyn && yyn <= YYLAST && yycheck[yyn] == YYSYMBOL_YYerror)
            {
              yyn = yytable[yyn];
              if (0 < yyn)
                break;
            }
        }

      /* Pop the current state because it cannot handle the error token.  */
      if (yyssp == yyss)
        YYABORT;


      yydestruct ("Error: popping",
                  YY_ACCESSING_SYMBOL (yystate), yyvsp);
      YYPOPSTACK (1);
      yystate = *yyssp;
      YY_STACK_PRINT (yyss, yyssp);
    }

  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END


  /* Shift the error token.  */
  YY_SYMBOL_PRINT ("Shifting", YY_ACCESSING_SYMBOL (yyn), yyvsp, yylsp);

  yystate = yyn;
  goto yynewstate;


/*-------------------------------------.
| yyacceptlab -- YYACCEPT comes here.  |
`-------------------------------------*/
yyacceptlab:
  yyresult = 0;
  goto yyreturnlab;


/*-----------------------------------.
| yyabortlab -- YYABORT comes here.  |
`-----------------------------------*/
yyabortlab:
  yyresult = 1;
  goto yyreturnlab;


/*-----------------------------------------------------------.
| yyexhaustedlab -- YYNOMEM (memory exhaustion) comes here.  |
`-----------------------------------------------------------*/
yyexhaustedlab:
  yyerror (YY_("memory exhausted"));
  yyresult = 2;
  goto yyreturnlab;


/*----------------------------------------------------------.
| yyreturnlab -- parsing is finished, clean up and return.  |
`----------------------------------------------------------*/
yyreturnlab:
  if (yychar != YYEMPTY)
    {
      /* Make sure we have latest lookahead translation.  See comments at
         user semantic actions for why this is necessary.  */
      yytoken = YYTRANSLATE (yychar);
      yydestruct ("Cleanup: discarding lookahead",
                  yytoken, &yylval);
    }
  /* Do not reclaim the symbols of the rule whose action triggered
     this YYABORT or YYACCEPT.  */
  YYPOPSTACK (yylen);
  YY_STACK_PRINT (yyss, yyssp);
  while (yyssp != yyss)
    {
      yydestruct ("Cleanup: popping",
                  YY_ACCESSING_SYMBOL (+*yyssp), yyvsp);
      YYPOPSTACK (1);
    }
#ifndef yyoverflow
  if (yyss != yyssa)
    YYSTACK_FREE (yyss);
#endif

  return yyresult;
}

#line 304 "frontend.y"


DataNode* createDataNode(int rm_dtype, bool isDirected, bool isWeighted, pair<int,int> infoDims, bool levelIndependence, string name, NumTypes indexType, NumTypes edgeType, NumTypes valueType){
    DataInfo* info = new DataInfo(RM_DTYPE, isDirected, isWeighted);
    info->setDims(infoDims.first, infoDims.second);
    DataLevel* level = new DataLevel(info, levelIndependence);
    DataNode* data = new DataNode(name, indexType, edgeType, valueType, level);
    return data;
}
DataNode* addDegrees_CIR(DataNode* graphData, TrainingLoopNode* trainingLoop){
    if (debug == 2) cout << "degrees\n";
	ForwardNode* onesTensorOp = new ForwardNode(POINTWISE, ONES_OP);
    DataNode* onesData = createDataNode(RM_DTYPE, false, false, {-1, 1}, true, "ones", INT32, INT32, F32);
	onesTensorOp->addOutputData(onesData);
	trainingLoop->addLoopNode(onesTensorOp);
	//* Dependencies
	RelationEdge* onesTensorGraphAssociation = new RelationEdge(graphData, ALL_RELATION, onesData, ROWS_RELATION);
	associations.push_back(onesTensorGraphAssociation);

	// The actual degrees calculation
	ForwardNode* degreesOp = new ForwardNode(AGGREGATE_NODE, AGGREGATE_MUL_SUM_DIRECT);
	DataInfo* degreesInfo = new DataInfo(RM_DTYPE);
	degreesInfo->setDims(-1, 1);
	DataLevel* rootDegreesLevel = new DataLevel(degreesInfo, true);
	DataNode* degreesData = new DataNode("degrees", INT32, INT32, F32, rootDegreesLevel);
	degreesOp->addInputData(onesData);
	degreesOp->addInputData(graphData);
	degreesOp->addOutputData(degreesData);
	degreesOp->addOpt(COARSE_COPT, 2);
	trainingLoop->addLoopNode(degreesOp);
	RelationEdge* degreesOpOnesDependency = new RelationEdge(onesData, ALL_RELATION, degreesData, ALL_RELATION);
	dependencies.push_back(degreesOpOnesDependency);
	RelationEdge* degreesOpGraphDependency = new RelationEdge(graphData, ALL_RELATION, degreesData, ROWS_RELATION);
	dependencies.push_back(degreesOpGraphDependency);
    return degreesData;
}
DataNode* addNormalization_CIR(DataNode* prevData, TrainingLoopNode* trainingLoop){
    if (debug == 2) cout << "normalization setup\n";
    ForwardNode* powerOp = new ForwardNode(POINTWISE, POWER_OP);
	powerOp->addParam("-0.5"); // hardcoded for now
    DataNode* normData = createDataNode(RM_DTYPE, false, false, {-1, 1}, true, "norm", INT32, INT32, F32);
	powerOp->addInputData(prevData);
	powerOp->addOutputData(normData);
	trainingLoop->addLoopNode(powerOp);
	RelationEdge* powerOpDegreesDependency = new RelationEdge(prevData, ALL_RELATION, normData, ALL_RELATION);
	dependencies.push_back(powerOpDegreesDependency);
    return normData;
}
DataNode* addNormCalc_CIR(DataNode* normData, DataNode* prevData, TrainingLoopNode* trainingLoop, int layerNum, bool featInput){ // prevData is either feat or res
    if (debug == 2) cout << "normalization-calculation\n";
	// 1st normalization calculation
	ForwardNode* normFeat1 = new ForwardNode(UPDATE_NODE, ROW_BROADCAST_OP);
    pair<int,int> normFeat1Data_inputDim = {-1, featInput ? m1.graph_transformations[FEAT_SIZE] : m1.output_input_classes};
    DataNode* normFeat1Data = createDataNode(RM_DTYPE, false, false, normFeat1Data_inputDim, true, "res", INT32, INT32, F32);
	normFeat1->addInputData(normData);
	normFeat1->addInputData(prevData);
	normFeat1->addOutputData(normFeat1Data);
	trainingLoop->addLoopNode(normFeat1);
	RelationEdge* normFeat1NormDependency = new RelationEdge(normData, ALL_RELATION, normFeat1Data, ROWS_RELATION);
	dependencies.push_back(normFeat1NormDependency);
	RelationEdge* normFeat1FeatDependency = new RelationEdge(prevData, ALL_RELATION, normFeat1Data, ALL_RELATION);
	dependencies.push_back(normFeat1FeatDependency);
	RelationEdge* normFeat1NormFeatAssociation = new RelationEdge(normData, ALL_RELATION, prevData, ROWS_RELATION);
	associations.push_back(normFeat1NormFeatAssociation);
    return normFeat1Data;

}
DataNode* addAggregate_CIR(DataNode* prevData, DataNode* graphData, TrainingLoopNode* trainingLoop, int layerNum){
    if (debug == 2) cout << "aggregate" << '\n';
    ForwardNode* aggregate = new ForwardNode(AGGREGATE_NODE, AGGREGATE_MUL_SUM_OP);
    pair<int,int> outputData_inputDim = {-1, (layerNum == 0) ? m1.graph_transformations[FEAT_SIZE] : m1.output_input_classes};
    DataNode* outputData = createDataNode(RM_DTYPE, false, false, outputData_inputDim, true, "res", INT32, INT32, F32);
    
    aggregate->addInputData(prevData);
    aggregate->addInputData(graphData); 
    aggregate->addOutputData(outputData);
    trainingLoop->addLoopNode(aggregate);

    // Relation (dependency) between features and aggregated output
    RelationEdge* inOutAggrRelationFeat = new RelationEdge(prevData, ALL_RELATION, outputData, ALL_RELATION);
    RelationEdge* inOutAggrRelationGraph = new RelationEdge(graphData, ALL_RELATION, outputData, ALL_RELATION);
    dependencies.push_back(inOutAggrRelationFeat);
    dependencies.push_back(inOutAggrRelationGraph);
    return outputData;

}
DataNode* addFFN_CIR(DataNode* prevData, TrainingLoopNode* trainingLoop, int layerNum){
    if (debug == 2) cout << "ffn" << '\n';
    ForwardNode* ffn = new ForwardNode(UPDATE_NODE, FFN_OP);
    // weight as matrix in DIR
    string weightNum = "weight" + to_string(layerNum+1);
    pair<int,int> weightInputDim;
    pair<int,int> resInputDim;
    if (layerNum == 0){
        weightInputDim = {m1.graph_transformations[FEAT_SIZE], m1.output_input_classes};
        resInputDim = {-1, m1.output_input_classes};
    }
    else{
        weightInputDim = {m1.output_input_classes, m1.graph_transformations[LABEL_SIZE]};
        resInputDim = {-1, m1.graph_transformations[LABEL_SIZE]};
    }
    DataNode* weightData = createDataNode(RM_DTYPE, false, false, weightInputDim, true, weightNum, INT32, INT32, F32);
    // Res DIR
    DataNode* resData = createDataNode(RM_DTYPE, false, false, resInputDim, true, "res", INT32, INT32, F32);
    ffn->addInputData(prevData);
    ffn->addInputData(weightData);
    ffn->addOutputData(resData);
    trainingLoop->addLoopNode(ffn);
    // Relation (dependency) between weight and features 
    RelationEdge* inOutWeightDepRelationFeat = new RelationEdge(prevData, ALL_RELATION, resData, ALL_RELATION);
    RelationEdge* inOutWeightDepRelationWeight = new RelationEdge(weightData, COLS_RELATION, resData, ROWS_RELATION);
    dependencies.push_back(inOutWeightDepRelationFeat);
    dependencies.push_back(inOutWeightDepRelationWeight);
    // Relation (association) between aggregate node and weight
    RelationEdge* inOutWeightAssociation = new RelationEdge(prevData, ROWS_RELATION, weightData, COLS_RELATION);
    associations.push_back(inOutWeightAssociation);
    return resData;

}
DataNode* addReLU_CIR(DataNode* prevData, TrainingLoopNode* trainingLoop, int layerNum){
    if (debug == 2) cout << "relu\n";
    // ReLU operation
	ForwardNode* reluOp = new ForwardNode(POINTWISE, NON_LNR_OP_RELU);
    pair<int,int> reluData_inputDim = {-1, (layerNum == 0) ? m1.output_input_classes : m1.graph_transformations[LABEL_SIZE]};
    DataNode* reluData = createDataNode(RM_DTYPE, false, false, reluData_inputDim, true, "res", INT32, INT32, F32);
	reluOp->addInputData(prevData);
	reluOp->addOutputData(reluData);
	trainingLoop->addLoopNode(reluOp);
	RelationEdge* reluOpOnesDependency = new RelationEdge(prevData, ALL_RELATION, reluData, ALL_RELATION);
	dependencies.push_back(reluOpOnesDependency);
    return reluData;
}

/*
purpose:
 - looks through model config layer_transformations and makes the appropriate nodes and edges accordingly
parameters:
 - isFirstLayer: stuff like degreesOp and powerOp are only created for first layer, if not first skip those 
 - connectingNodes: empty for first layer, but for future layers its the inputs to that layer
 - graph: either un-transformed or transformed graph to be put in as one of the connecting nodes
 - trainingLoop: where the nodes will be added
return:
 - output connecting nodes, whats fed in to the next layer (if there is one)
 */
DataNode* addLayer(int layerNum, DataNode* connectNode, DataNode* graphData, DataNode* featData, TrainingLoopNode* trainingLoop){
    DataNode* prevData = connectNode;
    for (LayerOpType t : m1.layer_operations){
        switch (t){
            case GET_DEGREES: // if first layer, this is first for gcn
                if (layerNum == 0){
                    prevData = addDegrees_CIR(graphData, trainingLoop);
                }
                break;
            case GET_NORMALIZATION:
                if (layerNum == 0){
                    normData = addNormalization_CIR(prevData, trainingLoop);
                }
                break;
            case MULT_NORM_RES:
                if (trainingLoop->getLoopNodeNum() < 5) // temporary fix to see if use featData or resData
                    prevData = addNormCalc_CIR(normData, featData, trainingLoop, layerNum, true);
                else
                    prevData = addNormCalc_CIR(normData, prevData, trainingLoop, layerNum, false);
                break;
            case MESSAGE_PASSING_AGGREGATE:
                prevData = addAggregate_CIR(prevData, graphData, trainingLoop, layerNum);
                break;
            case FEED_FORWARD_NN:
                prevData = addFFN_CIR(prevData, trainingLoop, layerNum);
                break;
            case NON_LINEARITY:
                prevData = addReLU_CIR(prevData, trainingLoop, layerNum);
                break;
        }
    }
    return prevData;
}
void generate_ir(){

    DataNode* graphData; 
    DataNode* featData;
    RelationEdge* graphFeatAssociation;
    if (m1.dataset_name != "\0"){ // load dataset
        if (debug == 2) cout << "load dataset section with name " << m1.dataset_name << "\n";
        auto loadDataset = ForwardNode(POINTWISE, LOAD_OP);
        loadDataset.addParam(m1.dataset_name);
        graphData = createDataNode(CSR_STYPE, true, true, {0,0}, true, "adj0", INT32, INT32, F32);
        featData = createDataNode(RM_DTYPE, false, false, {-1, -2}, true, "t_iden", INT32, INT32, F32);

        // association between graph and features
        graphFeatAssociation = new RelationEdge(graphData, ALL_RELATION, featData, ROWS_RELATION);
        associations.push_back(graphFeatAssociation);
        loadDataset.addOutputData(featData);
        loadDataset.addOutputData(graphData);

        program.push_back(&loadDataset);
    }
    bool createdTransformedGraph = false;
    DataNode* graph;
    if (m1.data_transformations.size() > 0){ // need a transformed graph to be made
        DataInfo* originalGraphInfo = dynamic_cast<DataInfo*>(graphData->getData()->next());
        DataInfo* transformedGraphInfo = new DataInfo(CSR_STYPE, !m1.graph_transformations[UNDIRECTED], !m1.graph_transformations[UNWEIGHTED]);
        if (m1.data_transformations[0].first == COL_TILE){ // only one data transform for now for gcn
            transformedGraphInfo->addOpt(COL_TILE_DOPT, m1.data_transformations[0].second);
        }
        DataLevel* transformedTileGraphLevel = new DataLevel(transformedGraphInfo, false);
	    DataLevel* transformedRootGraphLevel = new DataLevel(transformedTileGraphLevel, true);
	    DataNode* transformedGraph = new DataNode("graph_tile", graphData->getIType(), graphData->getNType(), graphData->getVType(), transformedRootGraphLevel);

        RelationEdge* trgraphFeatAssociation = new RelationEdge(transformedGraph, ALL_RELATION, featData, ROWS_RELATION);
        associations.push_back(trgraphFeatAssociation);
        if (m1.data_transformations[0].first == COL_TILE){ // only one data transform for now for gcn
            TransformData* tileTransformation = new TransformData(COL_TILE_DOPT);
            /* tileTransformation->addParam(m1.data_transformations[0].second); */
            tileTransformation->addParam("65000"); // why is it a string parameter?
            TransformEdge* graphTrgraph = new TransformEdge(graphData, transformedGraph);
            graphTrgraph->addTransformation(tileTransformation);
            transforms.push_back(graphTrgraph);
        }

        graph = transformedGraph;
    }
    else{ // then make sure to modify the original graph with the schedule transformations!
        
        DataInfo* graphInfo = dynamic_cast<DataInfo*>(graphData->getData()->next());
        graphInfo->setDirected(!m1.graph_transformations[UNDIRECTED]);
        graphInfo->setWeighted(!m1.graph_transformations[UNWEIGHTED]);
        graph = graphData;
    }

    DataInfo* featInfo = dynamic_cast<DataInfo*>(featData->getData()->next());
    featInfo->setDims(-1, m1.graph_transformations[FEAT_SIZE]);
    
    TrainingLoopNode* trainingLoop = new TrainingLoopNode(m1.iterations, CROSS_ENTROPY, ADAM, m1.validation_step);
    DataNode* connectNode;
    for (int i = 0; i < m1.num_layers; i++){
        connectNode = addLayer(i, connectNode, graph, featData, trainingLoop); 
    }
    program.push_back(trainingLoop);

    cout << "IR Generated!\n";
}

int main(int argc, char** argv){
    /* read from file instead of stdin */
    if (argc < 2){
        cout << "no filename provided";
        return 1;
    }
    const char* fileName= argv[1];
    FILE *myfile = fopen(fileName, "r");
    if (!myfile) {
        cout << "Invalid File" << endl;
        return -1;
    }
    yyin = myfile;
    yydebug = 0;
    debug = 1; // switch to 0 ifyw no print statements
    yyparse();
    cout << "Parsed!\n";
    if (debug){
        cout << " ---------------- printing model config ----------------------\n";
        cout << m1.to_string() << '\n';
        cout << "---------------------------------------------------------------\n";
    }
    generate_ir();
    if (debug){
        cout << " --------     checking generated ir output ------------ \n";
        cout << "PROGRAM (CIR Nodes): " << program.size() << '\n';
        for (int i = 0; i < program.size(); i++){

            /* ComputeNode* brruv = dynamic_cast<ComputeNode*>(program[i]); */
            /* cout << "     program node " << i << " with op and opType " << brruv->getOp() << ' ' << brruv->getOpType() << '\n'; */
            cout << "        program node " << i << "\n";
        }
        cout << "DEPENDENCIES " << dependencies.size() << '\n';
        for (int i = 0; i < dependencies.size(); i++){
            cout << "     dependency edge " << i << " with nodes " << dependencies[i]->getNode1()->getName() << ' ' << dependencies[i]->getNode2()->getName() << '\n';
        }
        cout << "ASSOCIATIONS " << associations.size() << '\n';
        for (int i = 0; i < associations.size(); i++){
            cout << "     associations edge " << i << " with nodes " << associations[i]->getNode1()->getName() << ' ' << associations[i]->getNode2()->getName() << '\n';
        }
        cout << "TRANSFORMS " << transforms.size() << '\n';
        for (int i = 0; i < transforms.size(); i++){
            cout << "     transform edge " << i << " with nodes " << transforms[i]->getNode1()->getName() << ' ' << transforms[i]->getNode2()->getName() << '\n';
        }
    }
    fclose(myfile);
}
void yyerror(const char *s){
    printf("Failed to parse: %s\n", s);
    /* halt program */
    exit(-1);
}
