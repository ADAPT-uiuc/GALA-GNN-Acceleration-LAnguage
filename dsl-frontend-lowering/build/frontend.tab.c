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
#include "ir/frontendIR.h"
using namespace std;

extern int yydebug;
extern int yylex();
extern int yyparse();
extern FILE *yyin;

void yyerror(const char *s);
vector<CIRNode*> program;
vector<RelationEdge*> dependencies;
vector<RelationEdge*> associations;
vector<TransformEdge*> transforms;
map<string, DataNode*> dataNodeMap;
map<string, ForwardNode*> computeNodeMap;
map<string, int> trainArgs;

#line 96 "build/frontend.tab.c"

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
  YYSYMBOL_COMMENT = 10,                   /* COMMENT  */
  YYSYMBOL_SET_UNWEIGHTED = 11,            /* SET_UNWEIGHTED  */
  YYSYMBOL_SET_UNDIRECTED = 12,            /* SET_UNDIRECTED  */
  YYSYMBOL_MODEL_W = 13,                   /* MODEL_W  */
  YYSYMBOL_EVAL = 14,                      /* EVAL  */
  YYSYMBOL_TRAIN = 15,                     /* TRAIN  */
  YYSYMBOL_LAYER = 16,                     /* LAYER  */
  YYSYMBOL_LOSS = 17,                      /* LOSS  */
  YYSYMBOL_OPTIMIZER = 18,                 /* OPTIMIZER  */
  YYSYMBOL_ITERS = 19,                     /* ITERS  */
  YYSYMBOL_VAL_STEP = 20,                  /* VAL_STEP  */
  YYSYMBOL_RMSE_LOSS = 21,                 /* RMSE_LOSS  */
  YYSYMBOL_ADAM_T = 22,                    /* ADAM_T  */
  YYSYMBOL_AGGR_INIT = 23,                 /* AGGR_INIT  */
  YYSYMBOL_FN_ARG = 24,                    /* FN_ARG  */
  YYSYMBOL_MUL_SUM = 25,                   /* MUL_SUM  */
  YYSYMBOL_DSL_FN = 26,                    /* DSL_FN  */
  YYSYMBOL_DSL_DOT = 27,                   /* DSL_DOT  */
  YYSYMBOL_FFN_OUT = 28,                   /* FFN_OUT  */
  YYSYMBOL_SIZE_FN = 29,                   /* SIZE_FN  */
  YYSYMBOL_RELAXNLN = 30,                  /* RELAXNLN  */
  YYSYMBOL_QUANT = 31,                     /* QUANT  */
  YYSYMBOL_GRAPH_ATTR = 32,                /* GRAPH_ATTR  */
  YYSYMBOL_FEAT_ATTR = 33,                 /* FEAT_ATTR  */
  YYSYMBOL_RELU = 34,                      /* RELU  */
  YYSYMBOL_LABEL_ATTR = 35,                /* LABEL_ATTR  */
  YYSYMBOL_RABBIT_REORDER_OP = 36,         /* RABBIT_REORDER_OP  */
  YYSYMBOL_SAMPLE_RANDOM_OP = 37,          /* SAMPLE_RANDOM_OP  */
  YYSYMBOL_COLTILE = 38,                   /* COLTILE  */
  YYSYMBOL_AGGR = 39,                      /* AGGR  */
  YYSYMBOL_FEAT_SIZE_ASSIGN = 40,          /* FEAT_SIZE_ASSIGN  */
  YYSYMBOL_LABEL_SIZE_ASSIGN = 41,         /* LABEL_SIZE_ASSIGN  */
  YYSYMBOL_COARSEN = 42,                   /* COARSEN  */
  YYSYMBOL_INTEGER = 43,                   /* INTEGER  */
  YYSYMBOL_FLOAT = 44,                     /* FLOAT  */
  YYSYMBOL_LBRACE = 45,                    /* LBRACE  */
  YYSYMBOL_RBRACE = 46,                    /* RBRACE  */
  YYSYMBOL_LSQBRA = 47,                    /* LSQBRA  */
  YYSYMBOL_RSQBRA = 48,                    /* RSQBRA  */
  YYSYMBOL_DOT = 49,                       /* DOT  */
  YYSYMBOL_COMMA = 50,                     /* COMMA  */
  YYSYMBOL_IF = 51,                        /* IF  */
  YYSYMBOL_ELSE = 52,                      /* ELSE  */
  YYSYMBOL_DO = 53,                        /* DO  */
  YYSYMBOL_WHILE = 54,                     /* WHILE  */
  YYSYMBOL_TR = 55,                        /* TR  */
  YYSYMBOL_FA = 56,                        /* FA  */
  YYSYMBOL_NOT = 57,                       /* NOT  */
  YYSYMBOL_AND = 58,                       /* AND  */
  YYSYMBOL_OR = 59,                        /* OR  */
  YYSYMBOL_NOTEQ = 60,                     /* NOTEQ  */
  YYSYMBOL_EQ = 61,                        /* EQ  */
  YYSYMBOL_GREATER = 62,                   /* GREATER  */
  YYSYMBOL_LESS = 63,                      /* LESS  */
  YYSYMBOL_GREATEREQ = 64,                 /* GREATEREQ  */
  YYSYMBOL_LESSEQ = 65,                    /* LESSEQ  */
  YYSYMBOL_PLUS = 66,                      /* PLUS  */
  YYSYMBOL_MINUS = 67,                     /* MINUS  */
  YYSYMBOL_MULTIPLY = 68,                  /* MULTIPLY  */
  YYSYMBOL_DIVIDE = 69,                    /* DIVIDE  */
  YYSYMBOL_FFN = 70,                       /* FFN  */
  YYSYMBOL_DATASET = 71,                   /* DATASET  */
  YYSYMBOL_NONLN = 72,                     /* NONLN  */
  YYSYMBOL_SENSEI_OP = 73,                 /* SENSEI_OP  */
  YYSYMBOL_INT = 74,                       /* INT  */
  YYSYMBOL_NEW = 75,                       /* NEW  */
  YYSYMBOL_NULL_KEY = 76,                  /* NULL_KEY  */
  YYSYMBOL_YYACCEPT = 77,                  /* $accept  */
  YYSYMBOL_program = 78,                   /* program  */
  YYSYMBOL_load_dataset = 79,              /* load_dataset  */
  YYSYMBOL_algorithm = 80,                 /* algorithm  */
  YYSYMBOL_statements = 81,                /* statements  */
  YYSYMBOL_statement = 82,                 /* statement  */
  YYSYMBOL_layers = 83,                    /* layers  */
  YYSYMBOL_layer_def = 84,                 /* layer_def  */
  YYSYMBOL_model = 85,                     /* model  */
  YYSYMBOL_model_def = 86,                 /* model_def  */
  YYSYMBOL_layer_inits = 87,               /* layer_inits  */
  YYSYMBOL_layer_init = 88,                /* layer_init  */
  YYSYMBOL_model_init = 89,                /* model_init  */
  YYSYMBOL_model_uses = 90,                /* model_uses  */
  YYSYMBOL_model_use = 91,                 /* model_use  */
  YYSYMBOL_gnn_op = 92,                    /* gnn_op  */
  YYSYMBOL_function = 93,                  /* function  */
  YYSYMBOL_update_op = 94,                 /* update_op  */
  YYSYMBOL_schedules = 95,                 /* schedules  */
  YYSYMBOL_schedule = 96,                  /* schedule  */
  YYSYMBOL_data_transform = 97,            /* data_transform  */
  YYSYMBOL_function_transform = 98,        /* function_transform  */
  YYSYMBOL_data_var = 99,                  /* data_var  */
  YYSYMBOL_function_init = 100,            /* function_init  */
  YYSYMBOL_semiring_op = 101,              /* semiring_op  */
  YYSYMBOL_op = 102,                       /* op  */
  YYSYMBOL_train_args = 103,               /* train_args  */
  YYSYMBOL_train_arg = 104,                /* train_arg  */
  YYSYMBOL_args = 105,                     /* args  */
  YYSYMBOL_arg = 106,                      /* arg  */
  YYSYMBOL_bool = 107,                     /* bool  */
  YYSYMBOL_string = 108                    /* string  */
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
#define YYLAST   216

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  77
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  32
/* YYNRULES -- Number of rules.  */
#define YYNRULES  69
/* YYNSTATES -- Number of states.  */
#define YYNSTATES  186

/* YYMAXUTOK -- Last valid token kind.  */
#define YYMAXUTOK   331


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
      75,    76
};

#if YYDEBUG
/* YYRLINE[YYN] -- Source line where rule number YYN was defined.  */
static const yytype_int16 yyrline[] =
{
       0,    71,    71,    86,   116,   117,   130,   141,   142,   157,
     217,   223,   229,   230,   233,   246,   248,   250,   251,   253,
     255,   257,   261,   262,   264,   265,   270,   277,   282,   288,
     294,   295,   300,   304,   309,   320,   329,   338,   349,   379,
     392,   395,   400,   405,   410,   416,   419,   422,   422,   422,
     422,   424,   425,   428,   433,   435,   437,   440,   441,   446,
     446,   450,   450,   452,   452,   456,   456,   461,   461,   462
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
  "LOAD", "LPAREN", "RPAREN", "SEMICOLON", "QUOTE", "COMMENT",
  "SET_UNWEIGHTED", "SET_UNDIRECTED", "MODEL_W", "EVAL", "TRAIN", "LAYER",
  "LOSS", "OPTIMIZER", "ITERS", "VAL_STEP", "RMSE_LOSS", "ADAM_T",
  "AGGR_INIT", "FN_ARG", "MUL_SUM", "DSL_FN", "DSL_DOT", "FFN_OUT",
  "SIZE_FN", "RELAXNLN", "QUANT", "GRAPH_ATTR", "FEAT_ATTR", "RELU",
  "LABEL_ATTR", "RABBIT_REORDER_OP", "SAMPLE_RANDOM_OP", "COLTILE", "AGGR",
  "FEAT_SIZE_ASSIGN", "LABEL_SIZE_ASSIGN", "COARSEN", "INTEGER", "FLOAT",
  "LBRACE", "RBRACE", "LSQBRA", "RSQBRA", "DOT", "COMMA", "IF", "ELSE",
  "DO", "WHILE", "TR", "FA", "NOT", "AND", "OR", "NOTEQ", "EQ", "GREATER",
  "LESS", "GREATEREQ", "LESSEQ", "PLUS", "MINUS", "MULTIPLY", "DIVIDE",
  "FFN", "DATASET", "NONLN", "SENSEI_OP", "INT", "NEW", "NULL_KEY",
  "$accept", "program", "load_dataset", "algorithm", "statements",
  "statement", "layers", "layer_def", "model", "model_def", "layer_inits",
  "layer_init", "model_init", "model_uses", "model_use", "gnn_op",
  "function", "update_op", "schedules", "schedule", "data_transform",
  "function_transform", "data_var", "function_init", "semiring_op", "op",
  "train_args", "train_arg", "args", "arg", "bool", "string", YY_NULLPTR
};

static const char *
yysymbol_name (yysymbol_kind_t yysymbol)
{
  return yytname[yysymbol];
}
#endif

#define YYPACT_NINF (-84)

#define yypact_value_is_default(Yyn) \
  ((Yyn) == YYPACT_NINF)

#define YYTABLE_NINF (-1)

#define yytable_value_is_error(Yyn) \
  0

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
static const yytype_int16 yypact[] =
{
      33,    35,    47,    48,    56,   -84,    65,    22,    48,    71,
     -84,    77,    25,   -84,    92,    96,    22,   -84,   -84,   -84,
       7,   -84,   100,   -84,   -84,   102,    98,     4,   103,   105,
     -21,   104,   -84,    23,   106,    70,    72,   -84,   113,    -3,
      24,   114,   116,   117,   101,   113,    85,   -84,    97,   118,
     119,   -84,   -84,   -84,   -84,   -84,   -84,   113,   -84,   115,
     121,    80,   120,   -84,   -84,   -84,   124,   128,     8,   -84,
     116,   123,   125,   -29,   126,    -1,   131,   113,   113,    87,
     129,   130,    68,   132,   -84,   134,   138,   108,   -84,   -84,
     -84,   113,   -84,    99,    93,    95,   107,    18,   -84,   122,
      15,    36,   -84,   -84,   136,   140,   141,   143,   -84,     0,
     -84,   109,   144,    16,   -84,   -84,   -84,   -84,   -84,   127,
     -84,   133,    40,    40,   110,   112,   111,     1,   137,   -84,
     -84,    13,   -84,   152,   113,   -84,   -84,   153,   155,   156,
     157,   -84,   158,   159,    74,   163,   -84,   -84,   -84,    17,
     160,   161,   162,   164,    14,   -84,   -84,   165,   167,   170,
     -84,    55,   -84,   -84,   -84,   -84,   -84,   171,   -84,   -84,
      11,   -84,   135,   139,   173,   169,   142,   145,   174,   -84,
     -84,   -84,   -84,    12,   175,   -84
};

/* YYDEFACT[STATE-NUM] -- Default reduction number in state STATE-NUM.
   Performed when YYTABLE does not specify something else to do.  Zero
   means the default is an error.  */
static const yytype_int8 yydefact[] =
{
       0,     0,     0,     4,     0,     1,     0,     0,     4,     0,
      12,     0,     0,    40,     0,     0,     2,    30,    32,    33,
       0,     5,     0,    13,     6,     0,     0,    40,     0,     0,
       0,     0,    25,     0,     0,     0,     0,    31,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    57,     0,     0,
       0,    27,     9,    47,    48,    49,    50,     0,    11,     0,
       0,     0,     0,    42,    41,    43,     0,     0,     0,    15,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    24,
       0,     0,     0,     0,    57,     0,     0,     0,    21,    69,
       3,     0,    10,     0,    65,    60,    62,    64,    58,     0,
       0,     0,    36,    37,     0,     0,     0,     0,    44,     0,
      57,     0,     0,     0,     7,    66,    59,    61,    63,     0,
      29,     0,     0,     0,     0,     0,     0,     0,     0,    51,
      26,     0,    46,     0,     0,    67,    68,     0,     0,     0,
       0,    17,     0,     0,     0,     0,    14,     8,    45,     0,
       0,     0,     0,     0,     0,    20,    57,     0,     0,     0,
      52,     0,    28,    35,    34,    38,    39,     0,    16,    18,
       0,    23,     0,     0,     0,     0,    53,    55,     0,    22,
      54,    56,    57,     0,     0,    19
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
     -84,   -84,   -84,   176,   -84,    50,   -84,   177,   -84,   -84,
     -84,   -84,   -84,   -84,   146,   -84,   -84,   -84,   -84,   172,
     -84,   -84,    -7,   -84,   -84,   -84,   -84,   -84,   -83,   -84,
      62,   -84
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_uint8 yydefgoto[] =
{
       0,     2,     3,     7,   131,     8,     9,    10,    24,    25,
     154,   169,    42,    69,    70,    31,    32,    51,    16,    17,
      18,    19,    97,    34,   133,    57,   144,   160,    75,    98,
     137,    44
};

/* YYTABLE[YYPACT[STATE-NUM]] -- What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule whose
   number is the opposite.  If YYTABLE_NINF, syntax error.  */
static const yytype_uint8 yytable[] =
{
      20,   109,    13,    13,    13,    33,    93,   126,   142,    20,
      45,    38,    86,    49,    13,    13,   145,   167,   175,   184,
      39,    91,   120,   130,   162,    13,    62,   127,    27,    63,
      64,    61,    65,    94,    94,    94,     1,    66,    73,     4,
      28,    28,    95,    95,    95,    94,    94,     5,    29,    50,
      79,     6,    30,    46,    95,    95,    39,    87,    27,   146,
     168,    11,    14,    15,    39,    39,    39,    39,   118,    12,
     100,   101,    39,   170,    22,    96,    96,    96,    29,   104,
     105,   157,    30,    26,   113,    39,   121,    96,    96,    53,
      54,    55,    56,   158,   159,   135,   136,    62,    35,   183,
      63,    64,    36,    65,    40,    41,   106,    43,    72,    47,
     107,    48,    52,    59,    58,    60,    13,    74,    67,    68,
      71,    76,    80,   112,    77,    78,    83,   149,    81,    82,
      84,    85,    89,    90,    92,    99,    39,   102,   103,   108,
     110,   111,   122,   115,   114,   116,   123,   124,   119,   125,
     129,   143,   132,   139,    33,   140,   141,   117,   128,   148,
     150,   134,   151,   152,   153,   156,   155,   161,   163,   164,
     165,   172,   166,   171,   173,   174,   178,   179,   176,     0,
     182,   147,   177,   185,    21,   138,    23,     0,    37,     0,
       0,     0,   180,     0,     0,   181,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    88
};

static const yytype_int16 yycheck[] =
{
       7,    84,     3,     3,     3,    12,     7,     7,     7,    16,
       6,     4,     4,    34,     3,     3,     3,     3,     7,     7,
      49,    50,     7,     7,     7,     3,    29,   110,     3,    32,
      33,    38,    35,    34,    34,    34,     3,    13,    45,     4,
      16,    16,    43,    43,    43,    34,    34,     0,    23,    70,
      57,     3,    27,    49,    43,    43,    49,    49,     3,    46,
      46,     5,    40,    41,    49,    49,    49,    49,    50,     4,
      77,    78,    49,   156,     3,    76,    76,    76,    23,    11,
      12,     7,    27,     6,    91,    49,    50,    76,    76,    66,
      67,    68,    69,    19,    20,    55,    56,    29,     6,   182,
      32,    33,     6,    35,     4,     3,    38,     9,     7,     6,
      42,     6,     8,    43,     8,    43,     3,    32,     4,     3,
       3,    24,     7,    15,     6,     6,     6,   134,     7,    49,
       6,     3,     9,     8,     8,     4,    49,     8,     8,     7,
       6,     3,     6,    50,    45,    50,     6,     6,    26,     6,
       6,    14,    25,    43,   161,    43,    45,    50,    49,     7,
       7,    28,     7,     7,     7,     6,     8,     4,     8,     8,
       8,     4,     8,     8,     4,     4,     3,     8,    43,    -1,
       6,   131,    43,     8,     8,   123,     9,    -1,    16,    -1,
      -1,    -1,    50,    -1,    -1,    50,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    70
};

/* YYSTOS[STATE-NUM] -- The symbol kind of the accessing symbol of
   state STATE-NUM.  */
static const yytype_int8 yystos[] =
{
       0,     3,    78,    79,     4,     0,     3,    80,    82,    83,
      84,     5,     4,     3,    40,    41,    95,    96,    97,    98,
      99,    80,     3,    84,    85,    86,     6,     3,    16,    23,
      27,    92,    93,    99,   100,     6,     6,    96,     4,    49,
       4,     3,    89,     9,   108,     6,    49,     6,     6,    34,
      70,    94,     8,    66,    67,    68,    69,   102,     8,    43,
      43,    99,    29,    32,    33,    35,    13,     4,     3,    90,
      91,     3,     7,    99,    32,   105,    24,     6,     6,    99,
       7,     7,    49,     6,     6,     3,     4,    49,    91,     9,
       8,    50,     8,     7,    34,    43,    76,    99,   106,     4,
      99,    99,     8,     8,    11,    12,    38,    42,     7,   105,
       6,     3,    15,    99,    45,    50,    50,    50,    50,    26,
       7,    50,     6,     6,     6,     6,     7,   105,    49,     6,
       7,    81,    25,   101,    28,    55,    56,   107,   107,    43,
      43,    45,     7,    14,   103,     3,    46,    82,     7,    99,
       7,     7,     7,     7,    87,     8,     6,     7,    19,    20,
     104,     4,     7,     8,     8,     8,     8,     3,    46,    88,
     105,     8,     4,     4,     4,     7,    43,    43,     3,     8,
      50,    50,     6,   105,     7,     8
};

/* YYR1[RULE-NUM] -- Symbol kind of the left-hand side of rule RULE-NUM.  */
static const yytype_int8 yyr1[] =
{
       0,    77,    78,    79,    80,    80,    80,    81,    81,    82,
      82,    82,    83,    83,    84,    85,    86,    87,    87,    88,
      89,    90,    91,    91,    92,    92,    93,    93,    94,    94,
      95,    95,    96,    96,    97,    97,    97,    97,    97,    98,
      99,    99,    99,    99,    99,   100,   101,   102,   102,   102,
     102,   103,   103,   104,   104,   104,   104,   105,   105,   106,
     106,   106,   106,   106,   106,   106,   106,   107,   107,   108
};

/* YYR2[RULE-NUM] -- Number of symbols on the right-hand side of rule RULE-NUM.  */
static const yytype_int8 yyr2[] =
{
       0,     2,     3,     7,     0,     2,     2,     0,     2,     4,
       6,     4,     1,     2,     9,     3,     9,     0,     2,     7,
       7,     2,     9,     7,     3,     1,     6,     2,     7,     4,
       1,     2,     1,     1,     9,     9,     5,     5,     9,     9,
       1,     3,     3,     3,     5,     7,     1,     1,     1,     1,
       1,     0,     2,     3,     4,     3,     4,     0,     2,     2,
       1,     2,     1,     2,     1,     1,     2,     1,     1,     3
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
#line 72 "frontend.y"
    {
        program.push_back((yyvsp[-2].forwardNode));
        program.push_back((yyvsp[-1].trainingLoopNode));
        /*
        if transformation exists, passed through schedules
        then to the first compute node in training loop add 
        transformed graph as input data
        */
        if (dataNodeMap.find("TrGraph") != dataNodeMap.end()){
            (yyvsp[-1].trainingLoopNode)->getNode(0)->addInputData(dataNodeMap["TrGraph"]);
            RelationEdge* inOutAggrRelationGraph = new RelationEdge(dataNodeMap["TrGraph"], ALL_RELATION, dataNodeMap["Output-Aggregate"], ALL_RELATION);
        }
    }
#line 1333 "build/frontend.tab.c"
    break;

  case 3: /* load_dataset: IDENTIFIER ASSIGN LOAD LPAREN string RPAREN SEMICOLON  */
#line 87 "frontend.y"
    {
        (yyval.forwardNode) = new ForwardNode(POINTWISE, LOAD_OP);
        (yyval.forwardNode)->addParam((yyvsp[-2].sval)); 
        // Graph
        DataInfo* graphInfo = new DataInfo(CSR_STYPE, false, false);
        DataLevel* rootGraphLevel = new DataLevel(graphInfo, true);
        DataNode* graphData = new DataNode("Graph", INT32, INT32, F32, rootGraphLevel);
        // Feat
        DataInfo* featInfo = new DataInfo(RM_DTYPE);
        featInfo->setDims(-1, -2); 
        DataLevel* rootFeatLevel = new DataLevel(featInfo, true);
        DataNode* featData = new DataNode("Feat", INT32, INT32, F32, rootFeatLevel);

        dataNodeMap["Graph"] = graphData;
        dataNodeMap["Feat"] = featData; // for future use (e.g. TrainingLoop is in another rule)

        // Relation (association) between graph and features
        RelationEdge* graphFeatAssociation = new RelationEdge(graphData, ALL_RELATION, featData, ROWS_RELATION);
        associations.push_back(graphFeatAssociation);

        (yyval.forwardNode)->addOutputData(featData);
        (yyval.forwardNode)->addOutputData(graphData);
        
        free((yyvsp[-6].sval));
        free((yyvsp[-2].sval));
    }
#line 1364 "build/frontend.tab.c"
    break;

  case 4: /* algorithm: %empty  */
#line 116 "frontend.y"
            { (yyval.trainingLoopNode) = NULL; }
#line 1370 "build/frontend.tab.c"
    break;

  case 5: /* algorithm: statement algorithm  */
#line 118 "frontend.y"
    {
        int iters = trainArgs.find("iters") != trainArgs.end() ? trainArgs["iters"] : 100;
        if ((yyvsp[0].trainingLoopNode) == NULL){
            (yyval.trainingLoopNode) = new TrainingLoopNode(iters); // default for this one
        }
        else{
            (yyval.trainingLoopNode) = (yyvsp[0].trainingLoopNode);
        }
        if ((yyvsp[-1].forwardNode) != NULL){
            (yyval.trainingLoopNode)->addLoopNode((yyvsp[-1].forwardNode));
        }
    }
#line 1387 "build/frontend.tab.c"
    break;

  case 6: /* algorithm: layers model  */
#line 131 "frontend.y"
    {              // so for now just one layer+model then schedules
        int iters = trainArgs.find("iters") != trainArgs.end() ? trainArgs["iters"] : 100;
        if ((yyvsp[-1].trainingLoopNode) != NULL){
            (yyval.trainingLoopNode) = (yyvsp[-1].trainingLoopNode);
        }
        else{
            (yyval.trainingLoopNode) = new TrainingLoopNode(iters);
        }
    }
#line 1401 "build/frontend.tab.c"
    break;

  case 7: /* statements: %empty  */
#line 141 "frontend.y"
             { (yyval.trainingLoopNode) = NULL; }
#line 1407 "build/frontend.tab.c"
    break;

  case 8: /* statements: statements statement  */
#line 143 "frontend.y"
    {
        int iters = trainArgs.find("iters") != trainArgs.end() ? trainArgs["iters"] : 100;
        if ((yyvsp[-1].trainingLoopNode)){
            (yyval.trainingLoopNode) = (yyvsp[-1].trainingLoopNode);
        }
        else{
            (yyval.trainingLoopNode) = new TrainingLoopNode(iters);
        }
        if ((yyvsp[0].forwardNode)){
            (yyval.trainingLoopNode)->addLoopNode((yyvsp[0].forwardNode));
        }
    }
#line 1424 "build/frontend.tab.c"
    break;

  case 9: /* statement: IDENTIFIER ASSIGN gnn_op SEMICOLON  */
#line 158 "frontend.y"
    {
        // TODO: add some code to verify the aggregate function name matches
        if (string((yyvsp[-1].sval)) == "aggregate"){ // aggregate operation
            (yyval.forwardNode) = new ForwardNode(AGGREGATE_NODE, MUL_SUM_OP);
            DataInfo* outputInfo = new DataInfo(RM_DTYPE);
            outputInfo->setDims(-1, -2); // -1=N=232965, the number of nodes in the graph
            DataLevel* rootOutputLevel = new DataLevel(outputInfo, true);
            DataNode* outputData = new DataNode("Out1", INT32, INT32, F32, rootOutputLevel);
            dataNodeMap["Output-Aggregate"] = outputData;
            
            (yyval.forwardNode)->addInputData(dataNodeMap["Feat"]);
            // below is supposed to be transformed graph, but not sure if it exists yet b/c
            // schedule is at bottom of file (so will be read by parser latere), so will 
            // just add both until a removeInputData() method is created?
            (yyval.forwardNode)->addInputData(dataNodeMap["Graph"]); 
            (yyval.forwardNode)->addInputData(outputData);
            computeNodeMap["aggregate"] = (yyval.forwardNode);

            // Relation (dependency) between features and aggregated output
            RelationEdge* inOutAggrRelationFeat = new RelationEdge(dataNodeMap["Feat"], ALL_RELATION, outputData, ALL_RELATION);
            RelationEdge* inOutAggrRelationGraph = new RelationEdge(dataNodeMap["Graph"], ALL_RELATION, outputData, ALL_RELATION);
            dependencies.push_back(inOutAggrRelationFeat);
            dependencies.push_back(inOutAggrRelationGraph);
            
        }
        else if (string((yyvsp[-1].sval)) == "ffn"){ // weight operation
            (yyval.forwardNode) = new ForwardNode(UPDATE_NODE, FFN_OP);
            // weight as matrix in DIR
            DataInfo* weightInfo = new DataInfo(RM_DTYPE);
            weightInfo->setDims(-2, -3); // -1=N=232965, the number of nodes in the graph
            DataLevel* weightLevel = new DataLevel(weightInfo, true);
            DataNode* weightData = new DataNode("Weight1", INT32, INT32, F32, weightLevel);
            dataNodeMap["Weight1"] = weightData;

            // Res DIR
            DataInfo* resInfo = new DataInfo(RM_DTYPE);
            resInfo->setDims(-1, -3); // -1=N=232965, the number of nodes in the graph
            DataLevel* rootResLevel = new DataLevel(resInfo, true);
            DataNode* resData = new DataNode("Res1", INT32, INT32, F32, rootResLevel);
            dataNodeMap["Res1"] = resData;
            (yyval.forwardNode)->addInputData(dataNodeMap["Output-Aggregate"]);
            (yyval.forwardNode)->addInputData(weightData);
            (yyval.forwardNode)->addOutputData(resData);

            // Relation (dependency) between weight and features 
            RelationEdge* inOutWeightDepRelationFeat = new RelationEdge(dataNodeMap["Output-Aggregate"], ALL_RELATION, resData, ALL_RELATION);
            RelationEdge* inOutWeightDepRelationWeight = new RelationEdge(weightData, COLS_RELATION, resData, ROWS_RELATION);
            dependencies.push_back(inOutWeightDepRelationFeat);
            dependencies.push_back(inOutWeightDepRelationWeight);
            // Relation (association) between aggregate node and weight
            RelationEdge* inOutWeightAssociation = new RelationEdge(dataNodeMap["Output-Aggregate"], ROWS_RELATION, weightData, COLS_RELATION);
            associations.push_back(inOutWeightAssociation);
        }
        else if (string((yyvsp[-1].sval)) == "relu"){
            (yyval.forwardNode) = NULL;
        }
        free((yyvsp[-3].sval));
        free((yyvsp[-1].sval));
    }
#line 1488 "build/frontend.tab.c"
    break;

  case 10: /* statement: IDENTIFIER ASSIGN IDENTIFIER DOT GRAPH_ATTR SEMICOLON  */
#line 218 "frontend.y"
    {
        (yyval.forwardNode) = NULL;
        free((yyvsp[-5].sval));
        free((yyvsp[-3].sval));
    }
#line 1498 "build/frontend.tab.c"
    break;

  case 11: /* statement: IDENTIFIER ASSIGN function_init SEMICOLON  */
#line 224 "frontend.y"
    {
        (yyval.forwardNode) = NULL;
        free((yyvsp[-3].sval));
    }
#line 1507 "build/frontend.tab.c"
    break;

  case 12: /* layers: layer_def  */
#line 229 "frontend.y"
                   { (yyval.trainingLoopNode) = (yyvsp[0].trainingLoopNode); }
#line 1513 "build/frontend.tab.c"
    break;

  case 13: /* layers: layers layer_def  */
#line 231 "frontend.y"
    {}
#line 1519 "build/frontend.tab.c"
    break;

  case 14: /* layer_def: IDENTIFIER ASSIGN LAYER LPAREN args RPAREN LBRACE statements RBRACE  */
#line 234 "frontend.y"
    {
        int iters = trainArgs.find("iters") != trainArgs.end() ? trainArgs["iters"] : 100;
        if ((yyvsp[-1].trainingLoopNode) != NULL){
            (yyval.trainingLoopNode) = (yyvsp[-1].trainingLoopNode);
        }
        else{
            (yyval.trainingLoopNode) = new TrainingLoopNode(iters);
        }
        free((yyvsp[-8].sval));
    }
#line 1534 "build/frontend.tab.c"
    break;

  case 15: /* model: model_def model_init model_uses  */
#line 246 "frontend.y"
                                        {}
#line 1540 "build/frontend.tab.c"
    break;

  case 16: /* model_def: IDENTIFIER ASSIGN MODEL_W LPAREN args RPAREN LBRACE layer_inits RBRACE  */
#line 248 "frontend.y"
                                                                                   {}
#line 1546 "build/frontend.tab.c"
    break;

  case 17: /* layer_inits: %empty  */
#line 250 "frontend.y"
              { (yyval.irNode) = NULL; }
#line 1552 "build/frontend.tab.c"
    break;

  case 18: /* layer_inits: layer_inits layer_init  */
#line 251 "frontend.y"
                             {}
#line 1558 "build/frontend.tab.c"
    break;

  case 19: /* layer_init: IDENTIFIER ASSIGN IDENTIFIER LPAREN args RPAREN SEMICOLON  */
#line 253 "frontend.y"
                                                                       { free((yyvsp[-6].sval)); free((yyvsp[-4].sval)); }
#line 1564 "build/frontend.tab.c"
    break;

  case 20: /* model_init: IDENTIFIER ASSIGN IDENTIFIER LPAREN args RPAREN SEMICOLON  */
#line 255 "frontend.y"
                                                                       { free((yyvsp[-6].sval)); free((yyvsp[-4].sval)); }
#line 1570 "build/frontend.tab.c"
    break;

  case 21: /* model_uses: model_use model_use  */
#line 257 "frontend.y"
                                 { (yyval.irNode) = NULL; }
#line 1576 "build/frontend.tab.c"
    break;

  case 22: /* model_use: IDENTIFIER ASSIGN IDENTIFIER DOT EVAL LPAREN args RPAREN SEMICOLON  */
#line 261 "frontend.y"
                                                                               { free((yyvsp[-8].sval)); free((yyvsp[-6].sval)); }
#line 1582 "build/frontend.tab.c"
    break;

  case 23: /* model_use: IDENTIFIER DOT TRAIN LPAREN train_args RPAREN SEMICOLON  */
#line 262 "frontend.y"
                                                              { free((yyvsp[-6].sval)); }
#line 1588 "build/frontend.tab.c"
    break;

  case 24: /* gnn_op: data_var op data_var  */
#line 264 "frontend.y"
                              { free((yyvsp[-2].sval)); free((yyvsp[0].sval)); }
#line 1594 "build/frontend.tab.c"
    break;

  case 25: /* gnn_op: function  */
#line 266 "frontend.y"
    {
        (yyval.sval) = (yyvsp[0].sval);
    }
#line 1602 "build/frontend.tab.c"
    break;

  case 26: /* function: IDENTIFIER LPAREN data_var COMMA data_var RPAREN  */
#line 271 "frontend.y"
    {
        (yyval.sval) = strdup("aggregate");
        free((yyvsp[-5].sval));
        free((yyvsp[-3].sval));
        free((yyvsp[-1].sval));
    }
#line 1613 "build/frontend.tab.c"
    break;

  case 27: /* function: DSL_DOT update_op  */
#line 278 "frontend.y"
    {
        (yyval.sval) = (yyvsp[0].sval);
    }
#line 1621 "build/frontend.tab.c"
    break;

  case 28: /* update_op: FFN LPAREN data_var COMMA FFN_OUT data_var RPAREN  */
#line 283 "frontend.y"
    {
        (yyval.sval) = strdup("ffn");
        free((yyvsp[-4].sval));
        free((yyvsp[-1].sval));
    }
#line 1631 "build/frontend.tab.c"
    break;

  case 29: /* update_op: RELU LPAREN data_var RPAREN  */
#line 289 "frontend.y"
    {
        (yyval.sval) = strdup("relu");
        free((yyvsp[-1].sval));
    }
#line 1640 "build/frontend.tab.c"
    break;

  case 30: /* schedules: schedule  */
#line 294 "frontend.y"
                     {}
#line 1646 "build/frontend.tab.c"
    break;

  case 31: /* schedules: schedules schedule  */
#line 296 "frontend.y"
    {

    }
#line 1654 "build/frontend.tab.c"
    break;

  case 32: /* schedule: data_transform  */
#line 301 "frontend.y"
    {

    }
#line 1662 "build/frontend.tab.c"
    break;

  case 33: /* schedule: function_transform  */
#line 305 "frontend.y"
    {
        
    }
#line 1670 "build/frontend.tab.c"
    break;

  case 34: /* data_transform: data_var ASSIGN data_var DOT SET_UNDIRECTED LPAREN bool RPAREN SEMICOLON  */
#line 310 "frontend.y"
    {
        // if transformed graph already exists, then modify that as well
        // always modify the original graph
        if (dataNodeMap.find("TrGraph") != dataNodeMap.end()){
            DataInfo* TrInfo = dynamic_cast<DataInfo*>(dataNodeMap["TrGraph"]->getData()->next());
            TrInfo->setDirected(!(yyvsp[-2].ival));
        }
        DataInfo* GraphInfo = dynamic_cast<DataInfo*>(dataNodeMap["Graph"]->getData()->next());
        GraphInfo->setDirected(!(yyvsp[-2].ival));
    }
#line 1685 "build/frontend.tab.c"
    break;

  case 35: /* data_transform: data_var ASSIGN data_var DOT SET_UNWEIGHTED LPAREN bool RPAREN SEMICOLON  */
#line 321 "frontend.y"
    {
        if (dataNodeMap.find("TrGraph") != dataNodeMap.end()){
            DataInfo* TrInfo = dynamic_cast<DataInfo*>(dataNodeMap["TrGraph"]->getData()->next());
            TrInfo->setWeighted(!(yyvsp[-2].ival));
        }
        DataInfo* GraphInfo = dynamic_cast<DataInfo*>(dataNodeMap["Graph"]->getData()->next());
        GraphInfo->setWeighted(!(yyvsp[-2].ival));
    }
#line 1698 "build/frontend.tab.c"
    break;

  case 36: /* data_transform: FEAT_SIZE_ASSIGN LPAREN INTEGER RPAREN SEMICOLON  */
#line 330 "frontend.y"
    {
        DataInfo* featInfo = dynamic_cast<DataInfo*>(dataNodeMap["Feat"]->getData()->next());
        DataInfo* outAggrInfo = dynamic_cast<DataInfo*>(dataNodeMap["Output-Aggregate"]->getData()->next());
        int dimRow = featInfo->getDimRow();
        featInfo->setDims(dimRow, atoi((yyvsp[-2].sval)));
        outAggrInfo->setDims(dimRow, atoi((yyvsp[-2].sval)));
        free((yyvsp[-2].sval));
    }
#line 1711 "build/frontend.tab.c"
    break;

  case 37: /* data_transform: LABEL_SIZE_ASSIGN LPAREN INTEGER RPAREN SEMICOLON  */
#line 339 "frontend.y"
    {
        DataInfo* featInfo = dynamic_cast<DataInfo*>(dataNodeMap["Feat"]->getData()->next());
        DataInfo* weightInfo = dynamic_cast<DataInfo*>(dataNodeMap["Weight1"]->getData()->next());
        DataInfo* resInfo = dynamic_cast<DataInfo*>(dataNodeMap["Res1"]->getData()->next());
        int featDimRow = featInfo->getDimRow();
        int featDimCol = featInfo->getDimCol();
        weightInfo->setDims(featDimCol, atoi((yyvsp[-2].sval)));
        resInfo->setDims(featDimRow, atoi((yyvsp[-2].sval)));
        free((yyvsp[-2].sval));
    }
#line 1726 "build/frontend.tab.c"
    break;

  case 38: /* data_transform: data_var ASSIGN data_var DOT COLTILE LPAREN INTEGER RPAREN SEMICOLON  */
#line 350 "frontend.y"
    {
        // actually creating new DIR
        DataLevel* originalRootGraphLevel = dataNodeMap["Graph"]->getData();
        // TODO: ask about using DataItem* b/c it is an abstract class, so should either be DataLevel or DataInfo?
        DataInfo* originalGraphInfo = dynamic_cast<DataInfo*>(originalRootGraphLevel->next());
        DataInfo* transformedGraphInfo = new DataInfo(CSR_STYPE, originalGraphInfo->getDirected(), originalGraphInfo->getWeighted());
        transformedGraphInfo->addOpt(COL_TILE_DOPT, atoi((yyvsp[-2].sval))); // TODO: change 65000 to match user input
        DataLevel* transformedTileGraphLevel = new DataLevel(transformedGraphInfo, false);
        DataLevel* transformedRootGraphLevel = new DataLevel(transformedTileGraphLevel, true);
        DataNode* transformedGraph = new DataNode("Graph-Tile", dataNodeMap["Graph"]->getIType(), dataNodeMap["Graph"]->getNType(),
            dataNodeMap["Graph"]->getVType(), transformedRootGraphLevel);

        dataNodeMap["TrGraph"] = transformedGraph;

        // Association between transformed graph and features
        RelationEdge* trGraphFeatAssociation = new RelationEdge(transformedGraph, ALL_RELATION, dataNodeMap["Feat"], ROWS_RELATION);
	    associations.push_back(trGraphFeatAssociation);
        // Transformation between original graph and new one
        TransformData* tileTransformation = new TransformData(COL_TILE_DOPT);
        tileTransformation->addParam((yyvsp[-2].sval));
        TransformEdge* graphTrgraph = new TransformEdge(dataNodeMap["Graph"], transformedGraph);
        graphTrgraph->addTransformation(tileTransformation);
        transforms.push_back(graphTrgraph);
            
        RelationEdge* inOutAggrRelationTrGraph = new RelationEdge(transformedGraph, ALL_RELATION, dataNodeMap["Output-Aggregate"], ALL_RELATION);
        dependencies.push_back(inOutAggrRelationTrGraph);
        free((yyvsp[-6].sval));
    }
#line 1759 "build/frontend.tab.c"
    break;

  case 39: /* function_transform: data_var ASSIGN data_var DOT COARSEN LPAREN INTEGER RPAREN SEMICOLON  */
#line 380 "frontend.y"
    {
        if (computeNodeMap.find("aggregate") != computeNodeMap.end()){
            computeNodeMap["aggregate"]->addOpt(COARSE_COPT, atoi((yyvsp[-2].sval)));
        }
        else{
            cout << "error\n";
        }
        free((yyvsp[-2].sval));
    }
#line 1773 "build/frontend.tab.c"
    break;

  case 40: /* data_var: IDENTIFIER  */
#line 393 "frontend.y"
    {
    }
#line 1780 "build/frontend.tab.c"
    break;

  case 41: /* data_var: data_var DOT FEAT_ATTR  */
#line 396 "frontend.y"
    {
        (yyval.sval) = strdup("feats");
        free((yyvsp[-2].sval));
    }
#line 1789 "build/frontend.tab.c"
    break;

  case 42: /* data_var: data_var DOT GRAPH_ATTR  */
#line 401 "frontend.y"
    {
        (yyval.sval) = strdup("graphs");
        free((yyvsp[-2].sval));
    }
#line 1798 "build/frontend.tab.c"
    break;

  case 43: /* data_var: data_var DOT LABEL_ATTR  */
#line 406 "frontend.y"
    {
        (yyval.sval) = strdup("label");
        free((yyvsp[-2].sval));
    }
#line 1807 "build/frontend.tab.c"
    break;

  case 44: /* data_var: data_var DOT SIZE_FN LPAREN RPAREN  */
#line 411 "frontend.y"
    {
        (yyval.sval) = strdup("size");
        free((yyvsp[-4].sval));
    }
#line 1816 "build/frontend.tab.c"
    break;

  case 45: /* function_init: AGGR_INIT LPAREN FN_ARG ASSIGN DSL_FN semiring_op RPAREN  */
#line 417 "frontend.y"
    {}
#line 1822 "build/frontend.tab.c"
    break;

  case 46: /* semiring_op: MUL_SUM  */
#line 420 "frontend.y"
    {}
#line 1828 "build/frontend.tab.c"
    break;

  case 47: /* op: PLUS  */
#line 422 "frontend.y"
          {}
#line 1834 "build/frontend.tab.c"
    break;

  case 48: /* op: MINUS  */
#line 422 "frontend.y"
                     {}
#line 1840 "build/frontend.tab.c"
    break;

  case 49: /* op: MULTIPLY  */
#line 422 "frontend.y"
                                   {}
#line 1846 "build/frontend.tab.c"
    break;

  case 50: /* op: DIVIDE  */
#line 422 "frontend.y"
                                               {}
#line 1852 "build/frontend.tab.c"
    break;

  case 51: /* train_args: %empty  */
#line 424 "frontend.y"
             { (yyval.irNode) = NULL; }
#line 1858 "build/frontend.tab.c"
    break;

  case 52: /* train_args: train_args train_arg  */
#line 426 "frontend.y"
    {}
#line 1864 "build/frontend.tab.c"
    break;

  case 53: /* train_arg: ITERS ASSIGN INTEGER  */
#line 429 "frontend.y"
    {
        trainArgs["iters"] = atoi((yyvsp[0].sval));
        free((yyvsp[0].sval));
    }
#line 1873 "build/frontend.tab.c"
    break;

  case 54: /* train_arg: ITERS ASSIGN INTEGER COMMA  */
#line 434 "frontend.y"
    { trainArgs["iters"] = atoi((yyvsp[-1].sval)); free((yyvsp[-1].sval)); }
#line 1879 "build/frontend.tab.c"
    break;

  case 55: /* train_arg: VAL_STEP ASSIGN INTEGER  */
#line 436 "frontend.y"
    { trainArgs["val_step"] = atoi((yyvsp[0].sval)); free((yyvsp[0].sval)); }
#line 1885 "build/frontend.tab.c"
    break;

  case 56: /* train_arg: VAL_STEP ASSIGN INTEGER COMMA  */
#line 438 "frontend.y"
    { trainArgs["val_step"] = atoi((yyvsp[-1].sval)); free((yyvsp[-1].sval)); }
#line 1891 "build/frontend.tab.c"
    break;

  case 57: /* args: %empty  */
#line 440 "frontend.y"
       { (yyval.irNode) = NULL; }
#line 1897 "build/frontend.tab.c"
    break;

  case 58: /* args: args arg  */
#line 442 "frontend.y"
    {

    }
#line 1905 "build/frontend.tab.c"
    break;

  case 59: /* arg: INTEGER COMMA  */
#line 446 "frontend.y"
                    { free((yyvsp[-1].sval)); }
#line 1911 "build/frontend.tab.c"
    break;

  case 60: /* arg: INTEGER  */
#line 447 "frontend.y"
    {
        free((yyvsp[0].sval));
    }
#line 1919 "build/frontend.tab.c"
    break;

  case 62: /* arg: NULL_KEY  */
#line 451 "frontend.y"
    {}
#line 1925 "build/frontend.tab.c"
    break;

  case 63: /* arg: data_var COMMA  */
#line 452 "frontend.y"
                     { free((yyvsp[-1].sval)); }
#line 1931 "build/frontend.tab.c"
    break;

  case 64: /* arg: data_var  */
#line 453 "frontend.y"
    {
        free((yyvsp[0].sval));
    }
#line 1939 "build/frontend.tab.c"
    break;

  case 66: /* arg: RELU COMMA  */
#line 457 "frontend.y"
    {

    }
#line 1947 "build/frontend.tab.c"
    break;

  case 67: /* bool: TR  */
#line 461 "frontend.y"
          { (yyval.ival) = 1; }
#line 1953 "build/frontend.tab.c"
    break;

  case 68: /* bool: FA  */
#line 461 "frontend.y"
                           { (yyval.ival) = 2; }
#line 1959 "build/frontend.tab.c"
    break;

  case 69: /* string: QUOTE IDENTIFIER QUOTE  */
#line 463 "frontend.y"
    {
        (yyval.sval) = (char*) malloc(strlen((yyvsp[-1].sval)) + 2);
        sprintf((yyval.sval), "%s", (yyvsp[-1].sval));
        free((yyvsp[-1].sval));
    }
#line 1969 "build/frontend.tab.c"
    break;


#line 1973 "build/frontend.tab.c"

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

#line 470 "frontend.y"


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
    yyparse();
    cout << "PROGRAM (CIR Nodes): " << program.size() << '\n';
    cout << "DEPENDENCIES " << dependencies.size() << '\n';
    cout << "ASSOCIATIONS " << associations.size() << '\n';
    cout << "TRANSFORMS " << transforms.size() << '\n';

    fclose(myfile);
}
void yyerror(const char *s){
    printf("Failed to parse: %s\n", s);
    /* halt program */
    exit(-1);
}
